#include "QT_interface_Elise.h"

extern void NewSplit( const std::string  &  a2Stplit,std::string & aK0,std::vector<std::string>  & aSup);

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

    if (m_QTMainWindow->getEngine()->getData() != NULL)
        _data = m_QTMainWindow->getEngine()->getData();
    else
        _data = new cData;

    rebuildGlCamera();

    _data->computeCenterAndBBox();

    Init();

    connect(m_QTMainWindow,	SIGNAL(imagesAdded(int, bool)), this, SLOT(changeImages(int, bool)));

    connect(m_QTMainWindow,	SIGNAL(removePoint(QString)), this, SLOT(removePoint(QString)));

    connect(m_QTMainWindow,	SIGNAL(setName(QString)), this, SLOT(setAutoName(QString)));

    mAppli->SetInterface(this);

    // Table View :: begin      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ImagesSFModel*      proxyImageModel  = new ImagesSFModel(this);
    PointGlobalSFModel* proxyPointGlob   = new PointGlobalSFModel(this);
    //ObjectsSFModel*     proxyObjectModel = new ObjectsSFModel(this);

    proxyPointGlob->setSourceModel  (new ModelPointGlobal(0,mAppli));
    proxyImageModel->setSourceModel (new ModelCImage(0,mAppli));
    //proxyObjectModel->setSourceModel(new ModelObjects(0,mAppli));

    m_QTMainWindow->setModel(proxyPointGlob, proxyImageModel/*, proxyObjectModel*/);

    m_QTMainWindow->resizeTables();

    connect(((PointGlobalSFModel*)m_QTMainWindow->tableView_PG()->model())->sourceModel(),SIGNAL(pGChanged()), this, SLOT(rebuildGlPoints()));

    connect(this, SIGNAL(dataChanged()), proxyPointGlob, SLOT(invalidate()));

    connect(this, SIGNAL(dataChanged()), proxyImageModel, SLOT(invalidate()));

    // Table View   :: End        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    // Context Menu :: begin      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    connect(m_QTMainWindow->tableView_PG(),SIGNAL(customContextMenuRequested(const QPoint &)),this,SLOT(contextMenu_PGsTable(const QPoint &)));
    connect(m_QTMainWindow->tableView_Images(),SIGNAL(customContextMenuRequested(const QPoint &)),this,SLOT(contextMenu_ImagesTable(const QPoint &)));

    _menuPGView         = new QMenu(m_QTMainWindow);
    _menuImagesView     = new QMenu(m_QTMainWindow);

    _thisPointAction    = _menuPGView->addAction(tr("Change Images for this point"));

    QAction* deleteSelectedPGAction = _menuPGView->addAction(tr("Delete selected points"));
    QAction* validateSelectedPGAction = _menuPGView->addAction(tr("Validate selected points"));

    _thisImagesAction   = _menuImagesView->addAction(tr("View Images"));

     _signalMapperPG    = new QSignalMapper(this);

    connect(_signalMapperPG, SIGNAL(mapped(int)), this, SLOT(changeImagesPG(int)));
    connect(_thisPointAction, SIGNAL(triggered()), _signalMapperPG, SLOT(map()));
    connect(_thisImagesAction, SIGNAL(triggered()), this, SLOT(viewSelectImages()));
    connect(deleteSelectedPGAction, SIGNAL(triggered()), this, SLOT(deleteSelectedGlobalPoints()));
    connect(validateSelectedPGAction, SIGNAL(triggered()), this, SLOT(validateSelectedGlobalPoints()));

    // Context Menu :: End        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    connect(this,SIGNAL(dataChanged(bool, cSP_PointeImage*)), this, SLOT(rebuildGlPoints(bool, cSP_PointeImage*)));
    connect(this,SIGNAL(dataChanged(bool, cSP_PointeImage*)), this, SLOT(updateToolBar()));

    connect(m_QTMainWindow->tableView_PG(),SIGNAL(entered(QModelIndex)), this, SLOT(selectPointGlobal(QModelIndex)));

    connectDeviceElise(*m_QTMainWindow);
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
            mAppli->imageVis(idCImage(imageName))->CptAff() = -prio;
            prio--;
        }
    }
    changeImages(-4,true);
}

void cQT_Interface::deleteSelectedGlobalPoints()
{
    foreach (QString namePoint,_listSelectedPG)
        removePoint(namePoint);
}

void cQT_Interface::validateSelectedGlobalPoints()
{
    foreach (QString namePoint,_listSelectedPG)
        for (int iI = 0; iI < mAppli->nbImagesTot(); ++iI)
        {
               cImage* image = mAppli->imageTot(iI);
               cSP_PointeImage* spPI =  image->PointeOfNameGlobSVP(namePoint.toStdString());
               if(spPI && spPI->Saisie()->Etat() == eEPI_NonSaisi)

                   ChangeState(spPI,eEPI_Valide);
        }
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

    QModelIndexList indexList = m_QTMainWindow->tableView_PG()->selectionModel()->selectedIndexes();

    _listSelectedPG.clear();

    foreach (QModelIndex indexa, indexList)
        _listSelectedPG.push_back(model->data(model->index(indexa.row(), 0)).toString());

    _signalMapperPG->removeMappings(_thisPointAction);
    _signalMapperPG->setMapping(_thisPointAction, cVirtualInterface::idPointGlobal(pGName.toStdString()));
    _thisPointAction->setText(tr("Change images for ") + pGName);

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

//    qDebug() << "selModel->currentIndex().column() : " << selModel->currentIndex().column();
//    qDebug() << "selModel->currentIndex().row() : " << selModel->currentIndex().row();

    if (selModel->currentIndex().row() == -1 || selModel->currentIndex().column() == -1)
    {
        //qDebug() << "Nothing";
        return &mAppli->Interface()->GetCaseNamePoint(0);
    }
    else
    {
//        if(_cNamePt)
//            delete _cNamePt;

        string aName = selModel->currentIndex().data(Qt::DisplayRole).toString().toStdString();

        cCaseNamePoint* CNP = mAppli->Interface()->GetCaseNamePoint(aName);

        //cSP_PointGlob * aPt = mAppli->PGlobOfNameSVP(aName);
        if (CNP)

            if(CNP->mFree)
                _cNamePt = CNP;//new cCaseNamePoint(aName, eCaseSaisie); //fake pour faire croire à une saisie à la X11
            else
                _cNamePt = &mAppli->Interface()->GetCaseNamePoint(0);
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
			#ifdef USE_MIPMAP_HANDLER
            	if (m_QTMainWindow->getWidget(aK)->getGLData()->glImageMasked().cObjectGL::name() == aName)
			#else
            	if (m_QTMainWindow->getWidget(aK)->getGLData()->imageName() == aName)
			#endif
                return true;

    return false;
}

void cQT_Interface::SetInvisRef(bool aVal)
{
    mRefInvis = aVal;

    emit dataChanged(true);  //pour update des widgets (raccourci Ctrl+R)
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


// todo peut etre mis dans la partie metier
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
    return (eTypePts)m_QTMainWindow->getParams()->getPtCreationMode();
}

double cQT_Interface::PtCreationWindowSize()
{
    return m_QTMainWindow->getParams()->getPtCreationWindowSize();
}

void cQT_Interface::addPoint(QPointF point)
{
    if (m_QTMainWindow->currentWidget()->hasDataLoaded() && mAppli )
    {
        if(isPolygonZero())
            cVirtualInterface::addPoint(transformation(point),currentCImage());

        emit dataChanged(true);

        m_QTMainWindow->resizeTables();
    }

}

void cQT_Interface::removePointGlobal(cSP_PointGlob * pPg)
{
    ELISE_DEBUG_ERROR(pPg == NULL, "cQT_Interface::removePointGlobal", "pPg == NULL");

    if (pPg && mAppli)
    {
        QString namePG(pPg->PG()->Name().c_str());

        for (int i = 0; i < m_QTMainWindow->nbWidgets(); ++i)
        {
            cPolygon* rule =  m_QTMainWindow->getWidget(i)->getGLData()->polygon(1);
            if(rule)
            {
                if(rule->size() > 0 && rule->point(0).parent())
                {
                    if(namePG == rule->point(0).parent()->name())
                        rule->point(0).setParent(NULL);
                }

                if(rule->size() > 1 &&  rule->point(1).parent())
                {
                    if(namePG == rule->point(1).parent()->name())
                        rule->point(1).setParent(NULL);
                }
            }
        }
    }

    DeletePoint( pPg );
    emit dataChanged(true);
}

bool  cQT_Interface::isPolygonZero()
{
  return m_QTMainWindow->currentWidget()->getGLData()->getCurrentPolygonIndex() == 0;
}


void cQT_Interface::removePoint(QString aName)
{

	if (mAppli && isPolygonZero())
	{
		cSP_PointGlob *pPg = mAppli->PGlobOfNameSVP(aName.toStdString());
		if (pPg == NULL)
		{
			mAppli->Interface()->DeleteCaseNamePoint(aName.toStdString());
			emit dataChanged(true);
		}
		else
			removePointGlobal(mAppli->PGlobOfNameSVP(aName.toStdString()));
	}
}

void cQT_Interface::movePoint(int idPt)
{
    if( idPt >= 0 && mAppli)
    {
        cSP_PointeImage* aPIm = NULL;

        if(isPolygonZero())
        {
            aPIm = PointeImageInCurrentWGL(idPt);

            if(aPIm)
                UpdatePoints(aPIm, transformation(getGLPt_CurWidget(idPt)));
        }

        emit dataChanged(true, aPIm);
    }
}

void cQT_Interface::changeState(int state, int idPt)
{
    eEtatPointeImage aState = (eEtatPointeImage)state;

    if (aState!=eEPI_NonValue && idPt != -1 && mAppli && isPolygonZero())
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

            emit dataChanged(true, aPIm);
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

        if (!aPIm)
        {
            cout << "Pointe missing for " << oldName << endl;
            return;
        }
        else if(mAppli->ChangeName(oldName, newName))
        {
            emit dataChanged(true, aPIm);
        }
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
            {
                m_QTMainWindow->setDataToGLWidget(idPg == THISWIN ? CURRENT_IDW : aKW,getGlData(images[aKW]));
                images[aKW]->SetLoaded();
            }

            aKW++;
        }

        mAppli->OnModifLoadedImage();
        mAppli->SetImagesVis(images);

        rebuildGlPoints(true);
    }
}

void cQT_Interface::changeImages(int idPtGl, bool aUseCpt)
{
    changeImagesPG(idPointGlobal(idPtGl), aUseCpt);
}

void cQT_Interface::changeCurPose(void *widgetGL)
{
    if (((GLWidget*)widgetGL)->hasDataLoaded() && mAppli)
    {
        int idImg = idCImage(((GLWidget*)widgetGL)->getGLData());
        m_QTMainWindow->selectCameraIn3DP(idImg);
        emit dataChanged();
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

        m_QTMainWindow->selectPointAllWGL(!mAppli->PGlob(idPG) ? QString("") : namePointGlobal(idPG));
        rebuild3DGlPoints((cPointGlob*) (!mAppli->PGlob(idPG) ? NULL : mAppli->PGlob(idPG)->PG()));
    }
}

void cQT_Interface::selectPoint(int idPtCurGLW)
{
    if(isPolygonZero())
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
    return (*m_QTMainWindow->currentWidget()->getGLData()->polygon(0))[idPt];
}

string cQT_Interface::getNameGLPt_CurWidget(int idPt)
{
    return getGLPt_CurWidget(idPt).name().toStdString();
}

void cQT_Interface::cmdBascule()
{
    std::string aKeySsArb;
    std::vector<std::string> aVParams;
    NewSplit(mAppli->Param().KeyAssocOri().Val(),aKeySsArb,aVParams);
    QString oriName = QString(aVParams[0].c_str()).remove(0,1);

    QApplication *app = static_cast<QApplication *>(QApplication::instance());

    QFile file(app->applicationDirPath() + "/mm3d");

    if(file.exists())
        printf("%s\n",app->applicationDirPath().toStdString().c_str());

    std::string aNameExp = mAppli->DC()+StdPrefix(mAppli->Param().ExportPointeImage().Val());

    QProcess myProcess;

    // Start the QProcess instance.
    myProcess.execute("mm3d", QStringList()
                    << "vGCPBascule"
                    << ".*."
                    << oriName
                    << oriName + "Basculed"
                    << QString(aNameExp.c_str()).remove(0,2) + "-S3D.xml"
                    << QString(aNameExp.c_str()).remove(0,2) + "-S2D.xml" );

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

    emit dataChanged(true);
}

void cQT_Interface::filesDropped(const QStringList &filenames)
{
    m_QTMainWindow->loadPlyIn3DPrev(filenames,_data);
}

#ifdef USE_MIPMAP_HANDLER
	int cQT_Interface::idCImage(QString aNameImage)
	{
		ELISE_DEBUG_ERROR(mAppli == NULL, "cQT_Interface::idCImage", "mAppli == NULL");
		const string nameImage = aNameImage.toStdString();
		for (int i = 0; i < mAppli->nbImagesVis(); ++i)
		{
		   ELISE_DEBUG_ERROR(mAppli->imageVis(i) == NULL, "cQT_Interface::idCImage", "mAppli->imageVis(" << i << ") == NULL");
		   if (mAppli->imageVis(i)->Name() == nameImage) return i;
		}
		return -1;
	}
#else
	int cQT_Interface::idCImage(QString nameImage)
	{
		for (int i = 0; i < mAppli->nbImagesVis(); ++i)
		   if(mAppli->imageVis(i)->Name() == nameImage.toStdString())
		       return i;

		return -1;
	}
#endif

void cQT_Interface::toQVec3D(Pt3d<double> P, QVector3D& qP)
{
    qP.setX(P.x);
    qP.setY(P.y);
    qP.setZ(P.z);
}

QVector3D cQT_Interface::toQVec3D(Pt3d<double> P)
{
    QVector3D qP;
    qP.setX(P.x);
    qP.setY(P.y);
    qP.setZ(P.z);
    return qP;
}

void cQT_Interface::connectDeviceElise(SaisieQtWindow& win)
{
    win.setDevIOCamera((deviceIOCamera*)new deviceIOCameraElise);
    win.setDevIOImage((deviceIOImageElise*)new deviceIOImageElise);
    win.setDevIOTieFile((deviceIOTieFileElise*)new deviceIOTieFileElise);

    win.setBanniere(QString(getBanniereMM3D().c_str()));
    win.setGit_revision(QString(gitRevision().c_str()));
}

cImage * cQT_Interface::currentCImage()
{
    return cVirtualInterface::CImageVis(idCurrentCImage());
}

int cQT_Interface::getQTWinMode()
{
    return m_QTMainWindow->appMode();
}

void cQT_Interface::Warning(string aMsg)
{
    QMessageBox::warning(NULL, tr("Warning"), QString(aMsg.c_str()));
}

cImage *cQT_Interface::CImage(QString nameImage)
{
    return cVirtualInterface::CImageVis(idCImage(nameImage));
}

int cQT_Interface::idCurrentCImage()
{
    return idCImage(m_QTMainWindow->currentWidget()->getGLData());
}

int cQT_Interface::idCImage(cGLData* data)
{
    if(data)
		#ifdef USE_MIPMAP_HANDLER
		{
		    const QString &name = data->glImageMasked().cObjectGL::name();
		    return idCImage(name);
		}
        #else
			return idCImage(data->imageName());
		#endif
    else
        return -1;
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
    cImage* image = mAppli->imageVis(idCImage(idWGL));

    if(!image)
        return NULL;
    else
        return image->PointeOfNameGlobSVP(pg->Name());
}

void cQT_Interface::centerOnPtGlobal(int idWGL, cPointGlob* aPG)
{
    if(!m_QTMainWindow->getWidget(idWGL)->getGLData())
        return;

    cSP_PointeImage* ptI = pointeImage(aPG, idWGL);

    if(ptI && ptI->Visible() && ptI->Saisie())

            m_QTMainWindow->getWidget(idWGL)->centerViewportOnImagePosition(
                        transformation(ptI->Saisie()->PtIm(),idWGL),
                        m_QTMainWindow->currentWidget()->getZoom());
}

void cQT_Interface::HighlightPoint(cSP_PointeImage* aPIm)
{
    aPIm->Gl()->HighLighted() = !aPIm->Gl()->HighLighted();

    if(aPIm->Gl()->HighLighted() && aPIm->Gl()->PG()->P3D().IsInit())
    {
        const Pt3dr pt = aPIm->Gl()->PG()->P3D().Val();

        m_QTMainWindow->threeDWidget()->setTranslation(QVector3D(pt.x,pt.y,pt.z));
    }

}

cGLData * cQT_Interface::getGlData(int idWidget)
{
    return (idWidget == -1) ? m_QTMainWindow->currentWidget()->getGLData() : m_QTMainWindow->getWidget(idWidget)->getGLData();
}

#ifdef USE_MIPMAP_HANDLER
	cGLData *cQT_Interface::getGlData(cImage *image)
	{
		if(!image) return NULL;

		ELISE_DEBUG_ERROR(m_QTMainWindow->getEngine() == NULL, "cQT_Interface::getGlData", "m_QTMainWindow->getEngine() == NULL");
		cEngine &engine = *m_QTMainWindow->getEngine();
		
		QString nameImage = QString(image->Name().c_str());
		for (int iGd = 0; iGd < engine.nbGLData(); ++iGd)
		{
		    ELISE_DEBUG_ERROR(engine.getGLData(iGd) == NULL, "cQT_Interface::getGlData", "engine.getGLData(iGd) == NULL");
		    const QString &name = engine.getGLData(iGd)->glImageMasked().cObjectGL::name();
		    //~ if(nameImage == m_QTMainWindow->getEngine()->getGLData(iGd)->imageName())
		    if (nameImage == name) return engine.getGLData(iGd);
		}

		return NULL;
	}
#else
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
#endif

Pt2dr cQT_Interface::transformation(QPointF pt, int idImage)
{
    float scaleFactor = getGlData(idImage)->glImageMasked().getLoadedImageRescaleFactor();

	ELISE_DEBUG_ERROR(getGlData(idImage)->glImageMasked()._m_image == NULL, "cQT_Interface::transformation(1)", "getGlData(idImage)->glImageMasked()._m_image == NULL");

    return Pt2dr(pt.x()/scaleFactor,(getGlData(idImage)->glImageMasked()._m_image->height() - pt.y())/scaleFactor);
}

QPointF cQT_Interface::transformation(Pt2dr pt, int idImage)
{
    float scaleFactor = getGlData(idImage)->glImageMasked().getLoadedImageRescaleFactor();

	ELISE_DEBUG_ERROR(getGlData(idImage)->glImageMasked()._m_image == NULL, "cQT_Interface::transformation(2)", "getGlData(idImage)->glImageMasked()._m_image == NULL");

    return QPointF(pt.x*scaleFactor,getGlData(idImage)->glImageMasked()._m_image->height() - pt.y*scaleFactor);
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

   QPointF pt = transformation(aSom->PtIm(),idImag);
   QString name(aSom->NamePt().c_str());
   cPoint point(pt,name,true,aSom->Etat());

   m_QTMainWindow->getWidget(idImag)->addGlPoint(point, aPt1, aPt2, aPG->HighLighted(),0);
}

void cQT_Interface::rebuild3DGlPoints(cPointGlob * selectPtGlob)
{
    vector< cSP_PointGlob * > pGV = mAppli->PG();

    if(pGV.size())
    {
        _data->deleteCloud(0);

        GlCloud *cloud = new GlCloud();

        for (int i = 0; i < (int)pGV.size(); ++i)
        {
            cPointGlob * pg = pGV[i]->PG();

            if (pg != NULL && pg->P3D().IsInit() && (!pg->Disparu().IsInit() || (pg->Disparu().IsInit() && !pg->Disparu().Val())))
            {
                QColor colorPt = pGV[i]->HighLighted() ? Qt::red : Qt::green;

                Pt3dr pt = (pg->P3D().Val());

                QVector3D pt3D(pt.x,pt.y,pt.z);

                cloud->addVertex(GlVertex(pt3D, pg == selectPtGlob ? colorPt: Qt::blue));
            }
        }

        _data->addReplaceCloud(cloud);

        m_QTMainWindow->threeDWidget()->getGLData()->replaceCloud(_data->getCloud(0));

        m_QTMainWindow->threeDWidget()->resetView(false,false,false,false,true);
        m_QTMainWindow->option3DPreview();
    }
}

void cQT_Interface::updateToolBar()
{
    m_QTMainWindow->setTextToolBar(tr("3D length : ") + QString::number(lengthRule()));
}

float cQT_Interface::lengthRule()
{

    cPolygon* rule = m_QTMainWindow->currentWidget()->getGLData()->polygon(1);

    if(rule && rule->size() == 2)
    {
        cPoint* rPt0 = (cPoint*)rule->point(0).parent();
        cPoint* rPt1 = (cPoint*)rule->point(1).parent();

        if(rPt0 && rPt1)
        {

            int idPG0 = cVirtualInterface::idPointGlobal(rPt0->name().toStdString());
            int idPG1 = cVirtualInterface::idPointGlobal(rPt1->name().toStdString());

            if(idPG1>=0 &&  idPG0 >=0)
            {
                cSP_PointGlob* pG0 = mAppli->PGlob(idPG0);
                cSP_PointGlob* pG1 = mAppli->PGlob(idPG1);

                if(pG0 && pG1)
                {
                    Pt3dr *p3d0 = pG0->PG()->P3D().PtrVal();
                    Pt3dr *p3d1 = pG1->PG()->P3D().PtrVal();

                    QVector3D V(p3d1->x - p3d0->x,p3d1->y - p3d0->y,p3d1->z - p3d0->z);

                    return V.length();
                }
            }
        }
    }

    return 0.0;
}

void cQT_Interface::rebuildGlPoints(bool bSave, cSP_PointeImage* aPIm)
{
    rebuild2DGlPoints();

    rebuild3DGlPoints(aPIm);

    if (bSave) Save();
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
                const vector<cSP_PointeImage *> &  aVP = mAppli->imageVis(t)->VP();

                m_QTMainWindow->getWidget(i)->getGLData()->saveLockRule();

                m_QTMainWindow->getWidget(i)->getGLData()->polygon(0)->clear();

                for (int aK=0 ; aK<int(aVP.size()) ; aK++)

                    if (PtImgIsVisible(*(aVP[aK])))

                        addGlPoint(aVP[aK], i);

                m_QTMainWindow->getWidget(i)->getGLData()->applyLockRule();

                m_QTMainWindow->getWidget(i)->update();
            }
        }
    }
}

void cQT_Interface::rebuildGlCamera()
{
    for (int i = 0; i < mAppli->nbImagesVis(); ++i)
    {
        ElCamera * aCamera = mAppli->imageVis(i)->ElCaptCam();
        

        if (aCamera != NULL)
        {
            cCamHandlerElise * camElise = new cCamHandlerElise(aCamera->CS());
            _data->addCamera((cCamHandler*)camElise);
        }
    }
}
