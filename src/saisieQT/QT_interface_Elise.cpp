#include "QT_interface_Elise.h"

cQT_Interface::cQT_Interface(cAppli_SaisiePts &appli, SaisieQtWindow *QTMainWindow):
    m_QTMainWindow(QTMainWindow),
    _data(NULL),
    _aCpt(0),
    _currentPGlobal(NULL)
{
    _cNamePt = new cCaseNamePoint ("CHANGE", eCaseAutoNum);

    mParam = &appli.Param();
    mAppli = &appli;

    mRefInvis = appli.Param().RefInvis().Val();

    for (int aK = 0; aK < m_QTMainWindow->nbWidgets();++aK)
    {
        GLWidget* widget = m_QTMainWindow->getWidget(aK);

        connect(widget,	SIGNAL(addPoint(QPointF)), this,SLOT(addPoint(QPointF)));

        connect(widget,	SIGNAL(movePoint(int)), this,SLOT(movePoint(int)));

        connect(widget,	SIGNAL(selectPoint(int)), this,SLOT(selectPoint(int)));

        connect(widget,	SIGNAL(removePoint(int, int)), this,SLOT(changeState(int,int)));

        connect(widget,	SIGNAL(overWidget(void*)), this,SLOT(changeCurPose(void*)));

        connect(widget->contextMenu(),	SIGNAL(changeState(int,int)), this,SLOT(changeState(int,int)));

        connect(widget->contextMenu(),	SIGNAL(changeName(QString, QString)), this, SLOT(changeName(QString, QString)));

        connect(widget->contextMenu(),	SIGNAL(changeImagesSignal(int, bool)), this, SLOT(changeImages(int, bool)));
     }

    connect(m_QTMainWindow,	SIGNAL(showRefuted(bool)), this, SLOT(SetInvisRef(bool)));

    connect(m_QTMainWindow,	SIGNAL(undoSgnl(bool)), this, SLOT(undo(bool)));

    connect(m_QTMainWindow,	SIGNAL(sCloseAll()), this, SLOT(close()));

    connect(m_QTMainWindow->threeDWidget(),	SIGNAL(filesDropped(QStringList)), this, SLOT(filesDropped(QStringList)));

    _data = new cData;

    rebuildGlCamera();

    _data->computeBBox();

    m_QTMainWindow->init3DPreview(_data);

    Init();

    connect(m_QTMainWindow,	SIGNAL(imagesAdded(int, bool)), this, SLOT(changeImages(int, bool)));

    connect(m_QTMainWindow,	SIGNAL(removePoint(QString)), this, SLOT(removePoint(QString)));

    connect(m_QTMainWindow,	SIGNAL(setName(QString)), this, SLOT(setAutoName(QString)));

    mAppli->SetInterface(this);

    // Table View :: begin      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ImagesSFModel*      proxyImageModel = new ImagesSFModel(this);
    PointGlobalSFModel* proxyPointGlob  = new PointGlobalSFModel(this);

    proxyPointGlob->setSourceModel(new ModelPointGlobal(0,mAppli));
    proxyImageModel->setSourceModel(new ModelCImage(0,mAppli));

    m_QTMainWindow->setModel(proxyPointGlob,proxyImageModel);

    m_QTMainWindow->resizeTables();

    connect(((PointGlobalSFModel*)m_QTMainWindow->tableView_PG()->model())->sourceModel(),SIGNAL(pGChanged()), this, SLOT(rebuildGlPoints()));

    connect(this,SIGNAL(dataChanged()), proxyPointGlob, SLOT(invalidate()));

    connect(this,SIGNAL(dataChanged()), proxyImageModel, SLOT(invalidate()));

    // Table View   :: End        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    // Context Menu :: begin      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    connect(m_QTMainWindow->tableView_PG(),SIGNAL(customContextMenuRequested(const QPoint &)),this,SLOT(contextMenu_PGsTable(const QPoint &)));
    connect(m_QTMainWindow->tableView_Images(),SIGNAL(customContextMenuRequested(const QPoint &)),this,SLOT(contextMenu_ImagesTable(const QPoint &)));

    _menuPGView         = new QMenu(m_QTMainWindow);
    _menuImagesView     = new QMenu(m_QTMainWindow);

    _thisPointAction    = _menuPGView->addAction("Change Images for this point");
    _thisImagesAction   = _menuImagesView->addAction("View Images");

     _signalMapperPG    = new QSignalMapper(this);

    connect(_signalMapperPG, SIGNAL(mapped(int)), this, SLOT(changeImagesPG(int)));
    connect(_thisPointAction, SIGNAL(triggered()), _signalMapperPG, SLOT(map()));
    connect(_thisImagesAction, SIGNAL(triggered()), this, SLOT(viewSelectImages()));

    // Context Menu :: End        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    connect(this,SIGNAL(dataChanged(cSP_PointeImage*)), this, SLOT(rebuildGlPoints(cSP_PointeImage*)));

    connect(m_QTMainWindow->tableView_PG(),SIGNAL(entered(QModelIndex)), this, SLOT(selectPointGlobal(QModelIndex)));
}

void cQT_Interface::viewSelectImages()
{
    QAbstractItemModel* model = m_QTMainWindow->tableView_Images()->model();
    QModelIndexList indexList = m_QTMainWindow->tableView_Images()->selectionModel()->selectedIndexes();

    int prio = indexList.count();

    foreach (QModelIndex index, indexList)
    {
        if(!index.column())
        {
            QString imageName = model->data(model->index(index.row(), 0)).toString();
            mAppli->image(idCImage(imageName))->CptAff() = -prio;
            prio--;
        }
    }
    changeImages(-4,true);
}

void cQT_Interface::contextMenu_ImagesTable(const QPoint &widgetXY)
{
    Q_UNUSED(widgetXY);
    _menuImagesView->exec(QCursor::pos());
}

void cQT_Interface::contextMenu_PGsTable(const QPoint &widgetXY)
{
    Q_UNUSED(widgetXY);

    QModelIndex         index = m_QTMainWindow->tableView_PG()->currentIndex();
    QAbstractItemModel* model = m_QTMainWindow->tableView_PG()->model();
    QString             pGName= model->data(model->index(index.row(), 0)).toString();

    _signalMapperPG->removeMappings(_thisPointAction);
    _signalMapperPG->setMapping(_thisPointAction, cVirtualInterface::idPointGlobal(pGName.toStdString()));
    _thisPointAction->setText("Change images for " + pGName);

    _menuPGView->exec(QCursor::pos());
}

void cQT_Interface::Init()
{
    InitNbWindows();

    InitVNameCase();
}

void cQT_Interface::AddUndo(cOneSaisie *aSom)
{
    mAppli->AddUndo(*aSom, currentCImage());
}

cCaseNamePoint *cQT_Interface::GetIndexNamePoint()
{

    QItemSelectionModel *selModel = m_QTMainWindow->tableView_PG()->selectionModel();

    if (selModel->currentIndex().column() != 0)
    {
        return _cNamePt;
    }
    else
    {
        if(_cNamePt)
            delete _cNamePt;

        string aName = selModel->currentIndex().data(Qt::DisplayRole).toString().toStdString();

        cSP_PointGlob * aPt = mAppli->PGlobOfNameSVP(aName);
        if (!aPt)

            _cNamePt = new cCaseNamePoint(aName, eCaseSaisie); //fake pour faire croire à une saisie à la X11

        else
            _cNamePt = new cCaseNamePoint("CHANGE", eCaseAutoNum);
    }

    return _cNamePt;
}

bool cQT_Interface::isDisplayed(cImage* aImage)
{
    QString aName = QString(aImage->Name().c_str());

    for (int aK = 0; aK < m_QTMainWindow->nbWidgets();++aK)
        if (m_QTMainWindow->getWidget(aK)->hasDataLoaded())
            if (m_QTMainWindow->getWidget(aK)->getGLData()->imageName() == aName)
                return true;

    return false;
}

void cQT_Interface::SetInvisRef(bool aVal)
{
    mRefInvis = aVal;
}

void cQT_Interface::close()
{
    if(mAppli)
    {
        mAppli->Save();

        m_QTMainWindow->tableView_PG()->setModel(NULL);
        m_QTMainWindow->tableView_Images()->setModel(NULL);

        cGLData *glda = m_QTMainWindow->threeDWidget()->getGLData();
        m_QTMainWindow->threeDWidget()->reset();

        if(glda)
            delete glda;

        if(_data)
            delete _data;

        //cAppli_SaisiePts* appli = mAppli;
        mAppli = NULL;
        //delete appli;
    }
}

pair<int, string> cQT_Interface::IdNewPts(cCaseNamePoint *aCNP)
{
    int aCptMax = mAppli->GetCptMax() + 1;

    string aName = aCNP->mName;

    if (aCNP->mTCP == eCaseAutoNum)

        aName = nameFromAutoNum(aCNP, aCptMax);

    return pair<int,string>(aCptMax,aName);
}

eTypePts cQT_Interface::PtCreationMode()
{
    return m_QTMainWindow->getParams()->getPtCreationMode();
}

double cQT_Interface::PtCreationWindowSize()
{
    return m_QTMainWindow->getParams()->getPtCreationWindowSize();
}

void cQT_Interface::addPoint(QPointF point)
{
    if (m_QTMainWindow->currentWidget()->hasDataLoaded() && mAppli)
        if(cVirtualInterface::addPoint(transformation(point),currentCImage()))
        {
            emit dataChanged();
            m_QTMainWindow->resizeTables();
        }
}

void cQT_Interface::removePointGlobal(cSP_PointGlob * pPg)
{
    if (pPg && mAppli)
    {
        DeletePoint( pPg );
        emit dataChanged();
    }
}

void cQT_Interface::removePoint(QString aName)
{
    if (mAppli)
        removePointGlobal(mAppli->PGlobOfNameSVP(aName.toStdString()));
}

void cQT_Interface::movePoint(int idPt)
{
    if( idPt >= 0 && mAppli)
    {
        cSP_PointeImage* aPIm = PointeImageInCurrentWGL(idPt);

        if(aPIm)
        {
            UpdatePoints(aPIm, transformation(getGLPt_CurWidget(idPt)));

            emit dataChanged(aPIm);
        }
    }
}

void cQT_Interface::changeState(int state, int idPt)
{
    eEtatPointeImage aState = (eEtatPointeImage)state;

    if (aState!=eEPI_NonValue && idPt != -1 && mAppli)
    {
        cSP_PointeImage* aPIm = PointeImageInCurrentWGL(idPt);

        if (aPIm)
        {
            if (aState == eEPI_Disparu)

                removePointGlobal(aPIm->Gl());

            else
            {
                if(aState == eEPI_Highlight)

                    HighlightPoint(aPIm);

                else

                    ChangeState(aPIm, aState);

                for (int idWGL = 0; idWGL < m_QTMainWindow->nbWidgets(); ++idWGL)

                    centerOnPtGlobal(idWGL, aPIm->Gl()->PG());
            }

            emit dataChanged(aPIm);
        }
    }
}

void cQT_Interface::changeName(QString aOldName, QString aNewName)
{
    if(mAppli)
    {
        string oldName = aOldName.toStdString();
        string newName = aNewName.toStdString();

        cSP_PointeImage * aPIm = currentCImage()->PointeOfNameGlobSVP(oldName);

        if (aPIm)

            return;

        else if(mAppli->ChangeName(oldName, newName))

            emit dataChanged(aPIm);

        else

            QMessageBox::critical(m_QTMainWindow, "Error", "Point already exists");
    }
}

void cQT_Interface::changeImagesPG(int idPg, bool aUseCpt)
{
    if(mAppli)
    {
        int aKW = 0;

        vector<cImage *> images = ComputeNewImagesPriority(mAppli->PGlob(idPg),aUseCpt);

        int max = (idPg == THISWIN) ? 1 : min(m_QTMainWindow->nbWidgets(),(int)images.size());

        while (aKW < max)
        {
            images[aKW]->CptAff() = _aCpt++;

            if (!isDisplayed(images[aKW]))

                m_QTMainWindow->SetDataToGLWidget(idPg == THISWIN ? CURRENT_IDW : aKW,getGlData(images[aKW]));

            aKW++;
        }

        mAppli->SetImages(images);

        rebuildGlPoints();
    }
}

void cQT_Interface::changeImages(int idPtGl, bool aUseCpt)
{
    changeImagesPG(idPointGlobal(idPtGl),aUseCpt);
}

void cQT_Interface::changeCurPose(void *widgetGL)
{
    if (((GLWidget*)widgetGL)->hasDataLoaded() && mAppli)
    {
        int idImg = idCImage(((GLWidget*)widgetGL)->getGLData());
        m_QTMainWindow->selectCameraIn3DP(idImg);
        emit dataChanged(); // TODO un peu fort
    }
}

void cQT_Interface::selectPointGlobal(int idPG)
{
    if( mAppli)
    {

        if(mAppli->PGlob(idPG))
        {
            setCurrentPGlobal(mAppli->PGlob(idPG));
            emit dataChanged();
            m_QTMainWindow->resizeTables();
        }

        m_QTMainWindow->SelectPointAllWGL(!mAppli->PGlob(idPG) ? QString("") : namePointGlobal(idPG));
        rebuild3DGlPoints((cPointGlob*) (!mAppli->PGlob(idPG) ? NULL : mAppli->PGlob(idPG)->PG()));
    }
}

void cQT_Interface::selectPoint(int idPtCurGLW)
{
    selectPointGlobal(idPointGlobal(idPtCurGLW));
}

void cQT_Interface::selectPointGlobal(QModelIndex modelIndex)
{

    selectPointGlobal(cVirtualInterface::idPointGlobal(m_QTMainWindow->tableView_PG()->model()->data(modelIndex).toString().toStdString()));
}

int cQT_Interface::idPointGlobal(int idSelectGlPoint)
{
    return idSelectGlPoint < 0 ? idSelectGlPoint : cVirtualInterface::idPointGlobal(getNameGLPt_CurWidget(idSelectGlPoint));
}

QString cQT_Interface::namePointGlobal(int idPtGlobal)
{
    return QString((((cSP_PointGlob * )(mAppli->PG()[idPtGlobal])))->PG()->Name().c_str());
}

cPoint cQT_Interface::getGLPt_CurWidget(int idPt)
{
    return (*m_QTMainWindow->currentWidget()->getGLData()->currentPolygon())[idPt];
}

string cQT_Interface::getNameGLPt_CurWidget(int idPt)
{
    return getGLPt_CurWidget(idPt).name().toStdString();
}
cSP_PointGlob *cQT_Interface::currentPGlobal() const
{
    return _currentPGlobal;
}

void cQT_Interface::setAutoName(QString name)
{
    mAppli->Param().NameAuto().SetVal( name.toStdString().c_str() );
}

void cQT_Interface::undo(bool aBool)
{
    if (aBool)
        mAppli->Undo();
    else
        mAppli->Redo();

    emit dataChanged();
}

void cQT_Interface::filesDropped(const QStringList &filenames)
{
    m_QTMainWindow->loadPlyIn3DPrev(filenames,_data);
}

int cQT_Interface::idCImage(QString nameImage)
{

    for (int i = 0; i < mAppli->nbImages(); ++i)
       if(mAppli->image(i)->Name() == nameImage.toStdString())
           return i;

    return -1;
}

cImage * cQT_Interface::currentCImage()
{
    return cVirtualInterface::CImage(idCurrentCImage());
}

cImage *cQT_Interface::CImage(QString nameImage)
{
    return cVirtualInterface::CImage(idCImage(nameImage));
}

int cQT_Interface::idCurrentCImage()
{
    return idCImage(m_QTMainWindow->currentWidget()->getGLData());
}

int cQT_Interface::idCImage(cGLData* data)
{
    return idCImage(data->imageName());
}

int cQT_Interface::idCImage(int idGlWidget)
{
    return idCImage(m_QTMainWindow->getWidget(idGlWidget)->getGLData());
}

cSP_PointeImage * cQT_Interface::PointeImageInCurrentWGL(int idPointGL)
{
    return currentCImage()->PointeOfNameGlobSVP(getNameGLPt_CurWidget(idPointGL));
}

cSP_PointGlob *cQT_Interface::PointGlobInCurrentWGL(int idPointGL)
{
    if(idPointGL >= 0)
        return  PointeImageInCurrentWGL(idPointGL)->Gl();
    else
        return NULL;
}

cSP_PointeImage * cQT_Interface::pointeImage(cPointGlob* pg, int idWGL)
{
    cImage* image = mAppli->image(idCImage(idWGL));

    if(!image)
        return NULL;
    else
        return image->PointeOfNameGlobSVP(pg->Name());
}

void cQT_Interface::centerOnPtGlobal(int idWGL, cPointGlob* aPG)
{
    cSP_PointeImage* ptI = pointeImage(aPG, idWGL);

    if(ptI && ptI->Visible() && ptI->Saisie())

            m_QTMainWindow->getWidget(idWGL)->centerViewportOnImagePosition(
                        transformation(ptI->Saisie()->PtIm(),idWGL),
                        m_QTMainWindow->currentWidget()->getZoom());
}

void cQT_Interface::HighlightPoint(cSP_PointeImage* aPIm)
{
    aPIm->Gl()->HighLighted() = !aPIm->Gl()->HighLighted();

    if(aPIm->Gl()->HighLighted())

        m_QTMainWindow->threeDWidget()->setTranslation(aPIm->Gl()->PG()->P3D().Val());

}

cGLData * cQT_Interface::getGlData(int idWidget)
{
    return (idWidget == -1) ? m_QTMainWindow->currentWidget()->getGLData() : m_QTMainWindow->getWidget(idWidget)->getGLData();
}

cGLData *cQT_Interface::getGlData(cImage *image)
{
    if(!image) return NULL;

    for (int iGd = 0; iGd < m_QTMainWindow->getEngine()->nbGLData(); ++iGd)
    {
        QString nameImage = QString(image->Name().c_str());
            if(nameImage == m_QTMainWindow->getEngine()->getGLData(iGd)->imageName())
                return m_QTMainWindow->getEngine()->getGLData(iGd);
    }

    return NULL;
}

Pt2dr cQT_Interface::transformation(QPointF pt, int idImage)
{
    return Pt2dr(pt.x(),getGlData(idImage)->glImage()._m_image->height() - pt.y());
}

QPointF cQT_Interface::transformation(Pt2dr pt, int idImage)
{
    return QPointF(pt.x,getGlData(idImage)->glImage()._m_image->height() - pt.y);
}

void cQT_Interface::addGlPoint(cSP_PointeImage * aPIm, int idImag)
{
    cOneSaisie *aSom = aPIm->Saisie();
    cSP_PointGlob* aPG = aPIm->Gl();

    QPointF aPt1(0.f,0.f);
    QPointF aPt2(0.f,0.f);

    if (aPG && aPG->HighLighted())
    {
        Pt2dr epi1, epi2;
        if (aPIm->BuildEpipolarLine(epi1, epi2))
        {
            aPt1 = transformation(epi1,idImag);
            aPt2 = transformation(epi2,idImag);
        }
    }

    m_QTMainWindow->getWidget(idImag)->addGlPoint(transformation(aSom->PtIm(),idImag), aSom, aPt1, aPt2, aPG->HighLighted());
}

void cQT_Interface::rebuild3DGlPoints(cPointGlob * selectPtGlob)
{

    vector< cSP_PointGlob * > pGV = mAppli->PG();

    if(pGV.size())
    {
        bool first = _data->getNbClouds() == 0;

        if(!first)
            delete _data->getCloud(0);

        GlCloud *cloud = new GlCloud();

        for (int i = 0; i < (int)pGV.size(); ++i)
        {
            cPointGlob * pg = pGV[i]->PG();

            QColor colorPt = pGV[i]->HighLighted() ? Qt::red : Qt::green;

            if (pg == selectPtGlob)
                colorPt = Qt::blue;

            cloud->addVertex(GlVertex(Pt3dr(pg->P3D().Val()), colorPt));
        }

        if(first)
            _data->addCloud(cloud);
        else
            _data->replaceCloud(cloud);

        _data->computeBBox();

        m_QTMainWindow->threeDWidget()->getGLData()->replaceCloud(_data->getCloud(0));
        m_QTMainWindow->threeDWidget()->resetView(first,false,first,true);
        m_QTMainWindow->option3DPreview();
    }
}

void cQT_Interface::rebuildGlPoints(cSP_PointeImage* aPIm)
{
    rebuild2DGlPoints();

    rebuild3DGlPoints(aPIm);

    Save(); // TODO trop de sauvegarde ---> même en selection ou over
}

void cQT_Interface::rebuild3DGlPoints(cSP_PointeImage* aPIm)
{
    rebuild3DGlPoints(aPIm ? aPIm->Gl()->PG() : NULL);
}

void cQT_Interface::rebuild2DGlPoints()
{
    for (int i = 0; i < m_QTMainWindow->nbWidgets(); ++i)
    {
        if(m_QTMainWindow->getWidget(i)->hasDataLoaded())
        {
            int t = idCImage(i);

            if(t!=-1)
            {
                const vector<cSP_PointeImage *> &  aVP = mAppli->image(t)->VP();

                m_QTMainWindow->getWidget(i)->getGLData()->clearPolygon();

                for (int aK=0 ; aK<int(aVP.size()) ; aK++)

                    if (PtImgIsVisible(*(aVP[aK])))

                        addGlPoint(aVP[aK], i);

                m_QTMainWindow->getWidget(i)->update();
            }
        }
    }
}

void cQT_Interface::rebuildGlCamera()
{
    for (int i = 0; i < mAppli->nbImages(); ++i)
    {
        ElCamera * aCamera = mAppli->image(i)->CaptCam();
        _data->addCamera(aCamera->CS());
    }
}
