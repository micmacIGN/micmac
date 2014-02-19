#include "QT_interface_Elise.h"

cQT_Interface::cQT_Interface(cAppli_SaisiePts &appli, MainWindow *QTMainWindow):
    m_QTMainWindow(QTMainWindow),
    _data(NULL)
{
    mParam = &appli.Param();
    mAppli = &appli;

    mRefInvis = appli.Param().RefInvis().Val();

    for (int aK = 0; aK < m_QTMainWindow->nbWidgets();++aK)
    {
        connect(m_QTMainWindow->getWidget(aK),	SIGNAL(addPoint(QPointF)), this,SLOT(addPoint(QPointF)));

        connect(m_QTMainWindow->getWidget(aK),	SIGNAL(movePoint(int)), this,SLOT(movePoint(int)));

        connect(m_QTMainWindow->getWidget(aK)->contextMenu(),	SIGNAL(changeState(int,int)), this,SLOT(changeState(int,int)));
    }

    _data = new cData;

    for (int i = 0; i < mAppli->nbImages(); ++i)
    {
        ElCamera * aCamera = mAppli->images(i)->CaptCam();
        _data->addCamera(aCamera->CS());
    }

    _data->computeBBox();

    m_QTMainWindow->threeDWidget()->setGLData(new cGLData(_data));
    m_QTMainWindow->threeDWidget()->setOption(cGLData::OpShow_BBox | cGLData::OpShow_Cams);
    m_QTMainWindow->threeDWidget()->setOption(cGLData::OpShow_Ball | cGLData::OpShow_Mess | cGLData::OpShow_BBox,false);
}

void cQT_Interface::SetInvisRef(bool aVal)
{
    mRefInvis = aVal;

    //TODO:
    /* for (int aKW=0 ; aKW < (int)mWins.size(); aKW++)
    {
        mWins[aKW]->BCaseVR()->SetVal(aVal);
        mWins[aKW]->Redraw();
        mWins[aKW]->ShowVect();
    }*/
}

cCaseNamePoint *cQT_Interface::GetIndexNamePoint()
{


   /* Video_Win aW = mMenuNamePoint->W();
    aW.raise();

    for (int aK=0 ; aK<int(mVNameCase.size()) ; aK++)
    {
        int aGr = (aK%2) ? 255 : 200 ;
        Pt2di aPCase(0,aK);
        mMenuNamePoint->ColorieCase(aPCase,aW.prgb()(aGr,aGr,aGr),1);
        cCaseNamePoint & aCNP = mVNameCase[aK];
        mMenuNamePoint->StringCase(aPCase,aCNP.mFree ?  aCNP.mName : "***" ,true);
    }

    Clik aClk = aW.clik_in();
    //aW.lower();

    Pt2di aKse = mMenuNamePoint->Pt2Case(Pt2di(aClk._pt));
    cCaseNamePoint * aRes =  &(mVNameCase[aKse.y]);

    if (! aRes->mFree) return 0;

    return aRes;*/

    return 0;
}

std::pair<int, string> cQT_Interface::IdNewPts(cCaseNamePoint *aCNP)
{
   int aCptMax = mAppli->GetCptMax() + 1;

   std::string aName = aCNP->mName;
   if (aCNP->mTCP == eCaseAutoNum)
   {
      std::string nameAuto = mParam->NameAuto().Val();
      aName = nameAuto + ToString(aCptMax);
      aCNP->mName = nameAuto + ToString(aCptMax+1);
   }

   if (aCNP->mTCP == eCaseSaisie)
   {
         //mWEnter->raise();
         //ELISE_COPY(mWEnter->all_pts(),P8COL::yellow,mWEnter->odisc());

         // std::cin >> aName ;
         //aName = mWEnter->GetString(Pt2dr(5,15),mWEnter->pdisc()(P8COL::black),mWEnter->pdisc()(P8COL::yellow));
         //mWEnter->lower();
   }

   //mMenuNamePoint->W().lower();

   // std::cout << "cAppli_SaisiePts::IdNewPts " << aCptMax << " " << aName << "\n";
   //std::pair aRes(
   return std::pair<int,std::string>(aCptMax,aName);

}

int cQT_Interface::cImageIdxFromName(QString nameImage)
{
    int t = -1;

    for (int i = 0; i < mAppli->nbImages(); ++i)
    {
       QString nameCImage(mAppli->images(i)->Name().c_str());
       if(nameCImage == nameImage)
           t = i;
    }

    return t;
}

void cQT_Interface::addPoint(QPointF point)
{
    Pt2dr aPGlob(point.x(),m_QTMainWindow->currentWidget()->getGLData()->glMaskedImage._m_image->height() - point.y());

    cCaseNamePoint aCNP("CHANGE",eCaseAutoNum);

    QString nameImage = m_QTMainWindow->currentWidget()->getGLData()->glMaskedImage.cObjectGL::name();

    int t = cImageIdxFromName(nameImage);

    //printf("name : %s : \n", getAppliMetier()->images(t)->Name().c_str());

    if(t != -1)
        mAppli->images(t)->CreatePGFromPointeMono(aPGlob,eNSM_Pts,-1,&aCNP);

    rebuildGlPoints();
}



string cQT_Interface::nameSelectPt(int idPt)
{
    std::string name =m_QTMainWindow->currentWidget()->getGLData()->m_polygon[idPt].name().toStdString();

    return name;
}

void cQT_Interface::movePoint(int idPt)
{
    if(idPt >= 0 )
    {
        int t = cImageIdxCurrent();

        cSP_PointeImage* aPIm = currentPointeImage(idPt);

        if(aPIm)
        {
            cImage* mCurIm = mAppli->images(t);
            mAppli->AddUndo(*(aPIm->Saisie()),mCurIm);

            aPIm->Saisie()->PtIm() = transformation(m_QTMainWindow->currentWidget()->getGLData()->m_polygon[idPt]);
            //Redraw();
            aPIm->Gl()->ReCalculPoints();

            rebuildGlPoints();

            mAppli->Sauv();
        }
    }
}



void cQT_Interface::changeState(int state, int idPt)
{

    //int idPt = m_QTMainWindow->currentWidget()->getGLData()->m_polygon.idx();

    eEtatPointeImage aState = (eEtatPointeImage)state;


    if (aState!=eEPI_NonValue && idPt != -1)
    {
        cSP_PointeImage* aPIm = currentPointeImage(idPt);

        if (aPIm)
        {
            mAppli->AddUndo(*(aPIm->Saisie()),currentCImage());
            aPIm->Saisie()->Etat() = aState;
            aPIm->Gl()->ReCalculPoints();

            rebuildGlPoints();

            mAppli->Sauv();
        }
    }
}

cSP_PointeImage * cQT_Interface::currentPointeImage(int idPoint)
{
    int t = cImageIdxCurrent();

    cSP_PointeImage * aPIm = mAppli->images(t)->PointeOfNameGlobSVP(nameSelectPt(idPoint));

    return aPIm;
}

cImage * cQT_Interface::currentCImage()
{
    int t = cImageIdxCurrent();
    cImage* mCurIm = mAppli->images(t);

    return mCurIm;
}

int cQT_Interface::cImageIdxCurrent()
{
    return cImageIdxFromGL(m_QTMainWindow->currentWidget()->getGLData());
}

int cQT_Interface::cImageIdxFromGL(cGLData* data)
{
    QString nameImage = data->glMaskedImage.cObjectGL::name();

    int t = cImageIdxFromName(nameImage);

    return t;
}

int cQT_Interface::cImageIdx(int idGl)
{
    return cImageIdxFromGL(m_QTMainWindow->getWidget(idGl)->getGLData());
}

cGLData * cQT_Interface::getGlData(int idImage)
{
    cGLData * data = (idImage == -1) ? m_QTMainWindow->currentWidget()->getGLData() : m_QTMainWindow->getWidget(idImage)->getGLData();

    return data;
}

Pt2dr cQT_Interface::transformation(QPointF pt, int idImage)
{
    Pt2dr newPt(pt.x(),getGlData(idImage)->glMaskedImage._m_image->height() - pt.y());
    return newPt;
}

QPointF cQT_Interface::transformation(Pt2dr pt, int idImage)
{
    QPointF newPt(pt.x,getGlData(idImage)->glMaskedImage._m_image->height() - pt.y);
    return newPt;
}

void cQT_Interface::addGlPoint(const cOneSaisie& aSom,  int i)
{

    Pt2dr aP = aSom.PtIm();

    eEtatPointeImage aState = aSom.Etat();

    m_QTMainWindow->getWidget(i)->addGlPoint(transformation(aP,i),QString(aSom.NamePt().c_str()), aState );
}

void cQT_Interface::rebuildGlPoints()
{
    for (int i = 0; i < m_QTMainWindow->nbWidgets(); ++i)
    {
        if(m_QTMainWindow->getWidget(i)->hasDataLoaded())
        {
            int t = cImageIdx(i);

            if(t!=-1)
            {
                const std::vector<cSP_PointeImage *> &  aVP = mAppli->images(t)->VP();

                m_QTMainWindow->getWidget(i)->getGLData()->clearPolygon();

                for (int aK=0 ; aK<int(aVP.size()) ; aK++)
                    //if (WVisible(*(aVP[aK])))
                    {
                        const cOneSaisie  & aSom = *(aVP[aK]->Saisie());
                        addGlPoint(aSom, i);
                    }

                m_QTMainWindow->getWidget(i)->update();
            }
        }
    }
    std::vector< cSP_PointGlob * > pGV = mAppli->PG();    

    if(pGV.size())
    {

        m_QTMainWindow->threeDWidget()->getGLData()->Clouds.clear();
        _data->clearClouds();

        GlCloud *cloud = new GlCloud();

        for (int i = 0; i < (int)pGV.size(); ++i)
            cloud->addVertex(GlVertex(pGV[i]->PG()->P3D().Val(),Qt::green));

        _data->addCloud(cloud);

        _data->computeBBox();
        m_QTMainWindow->threeDWidget()->getGLData()->setData(_data);
        m_QTMainWindow->threeDWidget()->resetView(true,false,true,true);
        m_QTMainWindow->threeDWidget()->setOption(cGLData::OpShow_BBox | cGLData::OpShow_Cams);
        m_QTMainWindow->threeDWidget()->setOption(cGLData::OpShow_Ball | cGLData::OpShow_Mess | cGLData::OpShow_BBox,false);
    }

}
