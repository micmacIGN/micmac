/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

#include "StdAfx.h"


//********************************************************************************

cCaseNamePoint::cCaseNamePoint(const std::string & aName, eTypeCasePt aTCP) :
    mName      (aName),
    mTCP       (aTCP),
    mFree      (true)
{
}

//********************************************************************************

void cVirtualInterface::OUT_Map()
{
    std::map<std::string,cCaseNamePoint *>::iterator ETE;

    for(ETE = mMapNC.begin(); ETE!=mMapNC.end(); ++ETE)
        cout << "mMapNC : " << ETE->first << endl;

    for(int i = 0; i < (int)mVNameCase.size(); ++i)
    {
        cCaseNamePoint CNP = ((cCaseNamePoint)mVNameCase[i]);
        cout << "mVNameCase : "<< CNP.mName << " " <<(CNP.mFree ? "free" : "No free") << endl;

    }
}

void cVirtualInterface::DeletePoint(cSP_PointGlob * aSG)
{
    aSG->SetKilled();

    ChangeFreeNamePoint(aSG->PG()->Name(), true);  
}

void cVirtualInterface::ComputeNbFen(Pt2di &pt, int aNbW)
{
    pt.x = round_up(sqrt(aNbW-0.01));
    pt.y = round_up((double(aNbW)-0.01)/pt.x);
}

void cVirtualInterface::InitNbWindows()
{
    const cSectionWindows & aSW = mParam->SectionWindows();
    mNb2W = aSW.NbFenIm().Val();

    mNbW = mNb2W.x * mNb2W.y;

    if (mAppli->nbImagesVis() < mNbW)
    {
        mNbW = mAppli->nbImagesVis();

        ComputeNbFen(mNb2W, mNbW);
    }
}

void cVirtualInterface::InitVNameCase()
{
    std::string aNameAuto = mParam->NameAuto().Val();

    if (aNameAuto != "NONE")
    {
        mVNameCase.push_back( cCaseNamePoint(aNameAuto+ToString(mAppli->GetCptMax()+1),eCaseAutoNum) );
    }

    for
            (
             std::list<std::string>::const_iterator itN = mParam->FixedName().begin();
             itN !=mParam->FixedName().end();
             itN++
             )
    {
        // const std::string aName = itN->c_str();
        std::vector<std::string> aNew = mAppli->ICNM()->StdGetVecStr(*itN);
        for (int aK=0 ; aK< (int)aNew.size(); aK++)
        {
            mVNameCase.push_back(cCaseNamePoint(aNew[aK],eCaseStd));
        }
    }

    for (int aK=0 ; aK<int(mVNameCase.size()); aK++)
    {
        mMapNC[mVNameCase[aK].mName] = & mVNameCase[aK];
    }

    for (int aK=0 ; aK< (int)mAppli->PG().size() ; aK++)
    {
        ChangeFreeNamePoint(mAppli->PG()[aK]->PG()->Name(),false);
    }
}

cSP_PointGlob *cVirtualInterface::addPoint(Pt2dr pt, cImage *curImg)
{
    cSP_PointGlob * PG = NULL;

    if(curImg)
    {
        eTypePts        aType   = PtCreationMode();
        double          aSz     = PtCreationWindowSize();
        Pt2dr           aPGlob  = FindPoint(curImg,pt,aType,aSz,0);
        cCaseNamePoint* aCNP    = GetIndexNamePoint();        

        if (aCNP && aCNP->mFree)
            PG = curImg->CreatePGFromPointeMono(aPGlob, aType, aSz, aCNP);

    }

    return PG;
}

int cVirtualInterface::idPointGlobal(std::string nameGP)
{
    int idPG = -1;

    vector < cSP_PointGlob * > vPG = mAppli->PG();
    for (int iPG = 0; iPG < (int)vPG.size(); ++iPG)
    {
        cSP_PointGlob * aPG  = vPG[iPG];

        if(aPG->PG()->Name() == nameGP)
            idPG = iPG;
    }

    return idPG;
}

const char *cVirtualInterface::cNamePointGlobal(int idPtGlobal)
{
    vector < cSP_PointGlob * > vPG = mAppli->PG();
    cSP_PointGlob * aPG  = vPG[idPtGlobal];

    return aPG->PG()->Name().c_str();
}

int cVirtualInterface::idPointGlobal(cSP_PointGlob *PG)
{
    int id = -1;
    for (int i = 0; i < (int)mAppli->PG().size(); ++i)
        if(mAppli->PG()[i] == PG)
            id = i;

    return id;
}

cImage *cVirtualInterface::CImageVis(int idCimg)
{
    if(idCimg < 0 || idCimg >= (int)mAppli->imagesVis().size())
        return NULL;
    else
        return mAppli->imageVis(idCimg);
}

vector<cImage *> cVirtualInterface::ComputeNewImagesPriority(cSP_PointGlob *pg,bool aUseCpt)
{

    mAppli->SetImagesPriority(pg, aUseCpt);
    vector<cImage *> images = mAppli->imagesVis();
    mAppli->SortImages(images);


    for (int aK=0 ; aK<int(mAppli->imagesVis().size()) ; aK++)
    {
         mAppli->imagesVis()[aK]->SetMemoLoaded();
    }

    return images;
}

void cVirtualInterface::ChangeFreeNamePoint(const std::string & aName, bool SetFree)
{
    std::map<std::string,cCaseNamePoint *>::iterator it = mMapNC.find(aName);
    if (it == mMapNC.end())
        return;
    if (it->second->mTCP == eCaseStd)
    {
        it->second->mFree = SetFree;
    }      
}

void cVirtualInterface::Save()
{
    mAppli->Save();
}

string cVirtualInterface::nameFromAutoNum(cCaseNamePoint *aCNP, int aCptMax)
{
    string nameAuto = mParam->NameAuto().Val();
    aCNP->mName = nameAuto + ToString(aCptMax+1);
    return nameAuto + ToString(aCptMax);
}

bool cVirtualInterface::Visible(eEtatPointeImage aState)
{
    return  ((aState!=eEPI_Refute) || !RefInvis())
            && (aState!=eEPI_Disparu);
}

void cVirtualInterface::ChangeState(cSP_PointeImage *aPIm, eEtatPointeImage aState)
{
    aPIm->Saisie()->Etat() = aState;

    AddUndo(aPIm->Saisie());

    aPIm->Gl()->ReCalculPoints();
}

void cVirtualInterface::UpdatePoints(cSP_PointeImage *aPIm, Pt2dr pt)
{
    aPIm->Saisie()->PtIm() = pt;
    Redraw();

    AddUndo(aPIm->Saisie());

    aPIm->Gl()->ReCalculPoints();
}

const Pt2dr cVirtualInterface::PtEchec (-100000,-10000);

Pt2dr cVirtualInterface::FindPoint(cImage* curIm, const Pt2dr & aPIm,eTypePts aType,double aSz,cPointGlob * aPG)
{
    Tiff_Im aTF = curIm->Tif();
    Pt2di aSzT = aTF.sz();

    int aRab = 5 + round_up(aSz);
    if ((aPIm.x <aRab) || (aPIm.y <aRab) || (aPIm.x >aSzT.x-aRab)|| (aPIm.y >aSzT.y-aRab))
        return PtEchec;


    Pt2di aMil  = mAppli->SzRech() / 2;
    Im2D_INT4 aImA = mAppli->ImRechAlgo();
    mAppli->DecRech() = round_ni(aPIm) - aMil;
    Pt2di aDec = mAppli->DecRech();
    ELISE_COPY
    (
        aImA.all_pts(),
        curIm->FilterImage(trans(aTF.in_proj(),aDec),aType,aPG),
        aImA.out()
    );
    ELISE_COPY
    (
        aImA.all_pts(),
        trans(aTF.in_proj(),aDec),
        //  mCurIm->FilterImage(trans(aTF.in_proj(),aDec),aType),
        mAppli->ImRechVisu().out()
    );


    if (aType==eNSM_Pts)
    {
       return aPIm;
    }



    Pt2dr aPosImInit = aPIm-Pt2dr(aDec);



    bool aModeExtre = (aType == eNSM_MaxLoc) ||  (aType == eNSM_MinLoc) || (aType==eNSM_GeoCube);
    bool aModeMax = (aType == eNSM_MaxLoc) ||  (aType==eNSM_GeoCube);


    if (aModeExtre)
    {
         aPosImInit = Pt2dr(MaxLocEntier(aImA,round_ni(aPosImInit),aModeMax,2.1));
         aPosImInit = MaxLocBicub(aImA,aPosImInit,aModeMax);

         return aPosImInit + Pt2dr(aDec);
    }

    return aPIm;
}

cCaseNamePoint *cVirtualInterface::GetCaseNamePoint(string name)
{
    std::map<std::string,cCaseNamePoint *>::iterator iT = mMapNC.find(name);
    if (iT == mMapNC.end()) return NULL;
    return iT->second;
}

bool cVirtualInterface::PtImgIsVisible(cSP_PointeImage &aPIm)
{

    const cOneSaisie  & aSom = *(aPIm.Saisie());
    eEtatPointeImage aState = aSom.Etat();

    return aPIm.Visible() && Visible(aState);

}

//********************************************************************************

cAppli_SaisiePts::cAppli_SaisiePts(cResultSubstAndStdGetFile<cParamSaisiePts> aP2, bool instanceInterface) :
    mParam      (*aP2.mObj),
    mInterface  (0),
    mICNM       (aP2.mICNM),
    mDC         (aP2.mDC),
    mShowDet    (mParam.ShowDet().Val()),
    mSzRech     (100,100),
    mImRechVisu (mSzRech.x,mSzRech.y),
    mImRechAlgo (mSzRech.x,mSzRech.y),
    mMasq3DVisib(0),
    mPIMsFilter   (0)
{
    if (mParam.Masq3DFilterVis().IsInit())
    {
       mMasq3DVisib = cMasqBin3D::FromSaisieMasq3d(mDC+mParam.Masq3DFilterVis().Val());
    }


    if (mParam.PIMsFilterVis().IsInit())
    {
       mPIMsFilter = cMMByImNM::FromExistingDirOrMatch(mParam.PIMsFilterVis().Val(),false,1.0,mICNM->Dir());
    }

    Tiff_Im::SetDefTileFile(100000);

    InitImages();
    InitInPuts();



#if (ELISE_X11)
    if(instanceInterface)
    {
        SetImagesPriority(0,false);
        SortImages(mImagesVis);
        mInterface = new cX11_Interface(*this);
        mInterface->Init();
        OnModifLoadedImage();
    }
#endif

}


cMMByImNM *  cAppli_SaisiePts::PIMsFilter ()
{
    return mPIMsFilter;
}

const Pt2di &  cAppli_SaisiePts::SzRech() const     { return mSzRech;     }
Pt2di &        cAppli_SaisiePts::DecRech()          { return mDecRech;    }

Im2D_INT4      cAppli_SaisiePts::ImRechVisu() const { return mImRechVisu; }
Im2D_INT4      cAppli_SaisiePts::ImRechAlgo() const { return mImRechAlgo; }

bool &         cAppli_SaisiePts::ShowDet()          { return mShowDet;    }

cParamSaisiePts & cAppli_SaisiePts::Param()        const { return mParam; }
const std::string     & cAppli_SaisiePts::DC()     const { return mDC;    }
cSetOfSaisiePointeIm  & cAppli_SaisiePts::SOSPI()        { return mSOSPI; }

cInterfChantierNameManipulateur * cAppli_SaisiePts::ICNM() const { return mICNM; }

void cAppli_SaisiePts::InitImages()
{
    std::list<std::string>  aListeNameIm = mICNM->StdGetListOfFile(mParam.SetOfImages(),1);

    for
            (
             std::list<std::string>::const_iterator itN=aListeNameIm.begin();
             itN!=aListeNameIm.end();
             itN++
             )
    {
        AddImageOfImOfName(*itN,true);
    }
    mImagesVis = mImagesTot;
    mNbImVis  = mImagesVis.size();

    mNameSauvPtIm = mDC + mParam.NamePointesImage().Val();
    mDupNameSauvPtIm = mNameSauvPtIm + ".dup";
    if (ELISE_fp::exist_file(mNameSauvPtIm))
    {
        mSOSPI = StdGetObjFromFile<cSetOfSaisiePointeIm>
                (
                    mNameSauvPtIm,
                    StdGetFileXMLSpec("ParamSaisiePts.xml"),
                    "SetOfSaisiePointeIm",
                    "SetOfSaisiePointeIm"
                    );
        for
        (
             std::list<cSaisiePointeIm>::iterator itS=mSOSPI.SaisiePointeIm().begin();
             itS != mSOSPI.SaisiePointeIm().end();
             itS++
        )
        {
            // std::cout << " GGGGGhUjj " << itS->NameIm() << " " <<  itS->OneSaisie().size() << "\n";
            AddImageOfImOfName(itS->NameIm(),false);
        }
    }
    mNbImTot = mImagesTot.size();
}

void   cAppli_SaisiePts::AddImageOfImOfName (const std::string & aName,bool Visualisable)
{
     if (! GetImageOfNameSVP(aName))
     {
            mImagesTot.push_back(new cImage(aName,*this,Visualisable));
            mMapNameIms[aName] = mImagesTot.back();
     }

}



cImage *  cAppli_SaisiePts::GetImageOfNameSVP(const std::string & aName)
{
    std::map<std::string,cImage *>::iterator iT = mMapNameIms.find(aName);
    if (iT == mMapNameIms.end()) return 0;
    return iT->second;
}

cSP_PointGlob *  cAppli_SaisiePts::PGlobOfNameSVP(const std::string & aName)
{
    std::map<std::string,cSP_PointGlob *>::iterator iT = mMapPG.find(aName);
    if (iT == mMapPG.end()) return 0;
    return iT->second;
}

cSP_PointGlob *cAppli_SaisiePts::PGlob(int id)
{
    if (id < 0 || id >= (int)mPG.size()) return NULL;
    return mPG[id];
}

void cAppli_SaisiePts:: ErreurFatale(const std::string & aName)
{
    std::cout << "Erreur, sortie de programme, resultats sauvegardes dans dup";
    std::cout << "\n";
    std::cout <<  "ER = " << aName << "\n";
    std::cout << "\n";
    exit(-1);
}


void  cAppli_SaisiePts::RenameIdPt(std::string & anId)
{
    std::string aPref = mParam.Prefix2Add2IdPt().Val();

    if (aPref=="") return;

    int  aCmp = anId.compare(0,aPref.size(),aPref);
    if (aCmp==0)  return;  // DEJA PREFIXE

    anId = aPref + anId;
    //  std::cout << "RenameIdPt [" << aPref << "] == [" << anId << "] " << aCmp << "\n";
}

cSP_PointGlob * cAppli_SaisiePts::AddPointGlob(cPointGlob aPG,bool OkRessuscite,bool Init,bool ReturnAlways)
{

    if (Init)
        RenameIdPt(aPG.Name());


    std::map<std::string,cSP_PointGlob *>::iterator iT = mMapPG.find(aPG.Name());
    if (iT == mMapPG.end())
    {
        mSPG.PointGlob().push_back(aPG);
        mPG.push_back(new cSP_PointGlob(*this,&(mSPG.PointGlob().back())));
        mMapPG[aPG.Name()] = mPG.back();
        // std::cout << "== APG CREAT3 "  << aPG.Disparu().ValWithDef(false) << "\n";
        return mPG.back();
    }
    /*
*/
    if (iT->second->PG()->Disparu().ValWithDef(false) && OkRessuscite)
    {
        if (! iT->second->PG()->FromDico().ValWithDef(false))
        {
            *(iT->second->PG()) = aPG;
        }

        iT->second->PG()->Disparu().SetNoInit();
        return iT->second;
    }
    if (ReturnAlways) return  iT->second;
    return 0;
}

void cAppli_SaisiePts::InitPG()
{

    mNameSauvPG = mDC + mParam.NamePointsGlobal().Val();
    mDupNameSauvPG = mNameSauvPG + ".dup";

    // std::cout << "TTttttcs::InitPG"  << mNameSauvPG << " " << ELISE_fp::exist_file(mNameSauvPG) << "\n";
    if (ELISE_fp::exist_file(mNameSauvPG))
    {
        cSetPointGlob aSPG = StdGetObjFromFile<cSetPointGlob>
                (
                    mNameSauvPG,
                    StdGetFileXMLSpec("ParamSaisiePts.xml"),
                    "SetPointGlob",
                    "SetPointGlob"
                    );

        for
                (
                 std::list<cPointGlob>::iterator itP=aSPG.PointGlob().begin();
                 itP!=aSPG.PointGlob().end();
                 itP++
                 )
        {
            if ( itP->Disparu().ValWithDef(false)  && (! itP->FromDico().ValWithDef(false)))
            {
            }
            else
            {
                AddPointGlob(*itP,false,true);
            }
        }
    }

    for
            (
             std::list<cImportFromDico>::iterator itIm=mParam.ImportFromDico().begin();
             itIm != mParam.ImportFromDico().end();
             itIm++
             )
    {
        cDicoAppuisFlottant aDic = StdGetObjFromFile<cDicoAppuisFlottant>
                (
                    mDC+itIm->File(),
                    StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                    "DicoAppuisFlottant",
                    "DicoAppuisFlottant"
                    );

        for
                (
                 std::list<cOneAppuisDAF>::iterator itA=aDic.OneAppuisDAF().begin();
                 itA != aDic.OneAppuisDAF().end();
                 itA++
                 )
        {
            cPointGlob aPG;
            aPG.Type() = itIm->TypePt();
            aPG.Name() = itA->NamePt() ;
            aPG.LargeurFlou().SetVal(itIm->LargeurFlou().Val());
            aPG.P3D().SetVal(itA->Pt());
            aPG.Incert().SetVal(itA->Incertitude());
            aPG.ContenuPt().SetNoInit();
            aPG.FromDico().SetVal(true);
            aPG.Pt3DFromDico().SetVal(itA->Pt());
            cSP_PointGlob * aNPG = AddPointGlob(aPG,false,true,true);

            if (mParam.FlouGlobEcras().Val())
                aNPG->PG()->LargeurFlou().SetVal(aPG.LargeurFlou().Val());
            if (mParam.TypeGlobEcras().Val())
                aNPG->PG()->Type() = aPG.Type();


        }

    }
}

void cAppli_SaisiePts::InitPointeIm()
{




    for
            (
             std::list<cSaisiePointeIm>::iterator itS=mSOSPI.SaisiePointeIm().begin();
             itS != mSOSPI.SaisiePointeIm().end();
             itS++
             )
    {
        static bool FirstNoIm = true;
        cImage * anIm = GetImageOfNameSVP(itS->NameIm());
        if (FirstNoIm && (!anIm)) // A priori ce warning va disparaitre ....
        {
            FirstNoIm = false;
            std::cout << "There is an image in Pointe with NO corresponding loaded image \n";
            std::cout << " First one is " << itS->NameIm() << "\n";
        }

        if (anIm)
        {
            anIm->SetSPIM(&(*itS));
            for
                    (
                     std::list<cOneSaisie>::iterator itOS=itS->OneSaisie().begin();
                     itOS!=itS->OneSaisie().end();
                     itOS++
                     )
            {
                if (itOS->Etat() != eEPI_Disparu)
                {
                    RenameIdPt(itOS->NamePt());
                    static bool FirstNoPG = true;
                    cSP_PointGlob * aPG = PGlobOfNameSVP(itOS->NamePt());
                    if (FirstNoPG && (!aPG))
                    {
                        FirstNoPG = false;
                        std::cout << "There is a 2D point in image with no global homologue \n";
                        std::cout << " First one is " <<  itOS->NamePt() << " in " << itS->NameIm() << "\n";
                    }
                    if (aPG)
                    {

                        anIm->AddAPointe(&(*itOS),aPG,true);
                    }
                }
            }
        }
    }

    for (std::vector<cSP_PointGlob*>::iterator itP=mPG.begin(); itP!=mPG.end() ; itP++)
    {
        AddPGInAllImages(*itP);
    }
}

void cAppli_SaisiePts::AddPGInAllImages(cSP_PointGlob  * aSPG)
{
    if (mParam.KeyAssocOri().IsInit())
    {
        Pt3dr aP3D(0,0,0);
        bool HasP3D = aSPG->Has3DValue() ;
        bool InMasq3D = true;
        if (HasP3D)
        {
            aP3D = aSPG->Best3dEstim();
            if (mMasq3DVisib)
            {
               InMasq3D = mMasq3DVisib->IsInMasq(aP3D);
            }
        }
        
        for (std::vector<cImage*>::iterator itI=mImagesTot.begin(); itI!=mImagesTot.end() ; itI++)
        {
            AddOnePGInImage(aSPG,**itI,HasP3D,aP3D,InMasq3D);
        }
    }
}

void cAppli_SaisiePts::AddOnePGInImage
     (cSP_PointGlob  * aSPG,cImage & anI,bool WithP3D,const Pt3dr & aP3d,bool InMasq3D)
{

    const cPointGlob & aPG = *(aSPG->PG());

    Pt2dr aPIm  = anI.PointArbitraire();
    bool OkInIm = InMasq3D;


    if ( OkInIm  && WithP3D)  
    {
        cCapture3D * aCapt3D = anI.Capt3d();
        if (aCapt3D)
        {
            aPIm =  aCapt3D->Ter2Capteur(aP3d); //  : anI.PointArbitraire();


            if (! aCapt3D->PIsVisibleInImage(aP3d))
            {
                OkInIm = false;
            }

            if (OkInIm && mMasq3DVisib)
            {
                ElSeg3D   aSeg = aCapt3D->Capteur2RayTer(aPIm);
                double anA = aSeg.AbscOfProj(aP3d);
                int aNb=50;
                for (int aK=aNb; (aK>=0) && (OkInIm) ; aK--)
                {
                    OkInIm = mMasq3DVisib->IsInMasq(aSeg.PtOfAbsc((anA*aK)/aNb));
                }
            }
        }
    }

    /// std::cout << "XccByyt "<< aSPG->PG()->Name() << " " << OkInIm << "\n";

    cSP_PointeImage * aPointeIm = anI.PointeOfNameGlobSVP(aPG.Name());

    if (aPointeIm)
    {
        if (aPointeIm->Saisie()->Etat()==eEPI_NonSaisi)
        {
            if ( OkInIm && anI.InImage(aPIm))
            {
                aPointeIm->Saisie()->PtIm() = aPIm;
                aPointeIm->Visible() = true;  // New MPD 13/01/15 , sinon evolue toujours dans le meme sens ??? 
            }
            else
            {
                aPointeIm->Visible() = false;
            }
        }
    }
    else
    {
        if (OkInIm && anI.InImage(aPIm))
        {
            cOneSaisie anOS;
            anOS.Etat() = eEPI_NonSaisi;
            anOS.NamePt() = aPG.Name();
            anOS.PtIm() = aPIm;
            anI.AddAPointe(&anOS,aSPG,false);
        }
    }
}

void cAppli_SaisiePts::GlobChangStatePointe
(
        const std::string & aName,
        const eEtatPointeImage aState
        )
{
    for
            (
             std::list<cSaisiePointeIm>::iterator itSPI=mSOSPI.SaisiePointeIm().begin();
             itSPI!=mSOSPI.SaisiePointeIm().end();
             itSPI++
             )
    {
        for
                (
                 std::list<cOneSaisie>::iterator itS=itSPI->OneSaisie().begin();
                 itS!=itSPI->OneSaisie().end();
                 itS++
                 )
        {
            if (itS->NamePt() == aName)
            {
                itS->Etat() = aState;
            }
        }
    }
}



void cAppli_SaisiePts::InitInPuts()
{
    //std::cout << "SPTS::CCCCC\n"; getchar();
    InitPG();
    //std::cout << "SPTS::DDDDDD\n"; getchar();
    InitPointeIm();

    // std::cout << "NB POINT GLOG " << mPG.size() << "\n";
    // Si on a change d'orientation, les points 3D ne sont plus valables ....
    for (int aKP=0 ; aKP<int(mPG.size())  ; aKP++)
    {
        mPG[aKP]->ReCalculPoints();
    }
    //std::cout << "SPTS::EEEEEE\n"; getchar();
    Save();
    //std::cout << "SPTS::FFFFF\n"; getchar();
}

cSetOfSaisiePointeIm PurgeSOSPI(const cSetOfSaisiePointeIm & aSOSPI)
{
    cSetOfSaisiePointeIm aRes;
    for
            (
             std::list<cSaisiePointeIm>::const_iterator itSPI=aSOSPI.SaisiePointeIm().begin();
             itSPI!=aSOSPI.SaisiePointeIm().end();
             itSPI++
             )
    {
        cSaisiePointeIm aSSP;
        aSSP.NameIm() = itSPI->NameIm();
        for
                (
                 std::list<cOneSaisie>::const_iterator itS=itSPI->OneSaisie().begin();
                 itS!=itSPI->OneSaisie().end();
                 itS++
                 )
        {
            if (
                    (itS->Etat() != eEPI_Disparu)
                    && (itS->Etat() != eEPI_NonValue)
                    )
            {
                aSSP.OneSaisie().push_back(*itS);
            }
        }
        aRes.SaisiePointeIm().push_back(aSSP);
    }
    return aRes;
}

void cAppli_SaisiePts::Save()
{
    cSetOfSaisiePointeIm aSOSPI = PurgeSOSPI(mSOSPI);
    MakeFileXML(aSOSPI,mDupNameSauvPtIm);
    MakeFileXML(aSOSPI,mNameSauvPtIm);

    MakeFileXML(mSPG,mDupNameSauvPG);
    MakeFileXML(mSPG,mNameSauvPG);

    if (mParam.ExportPointeImage().IsInit())
    {
        cSetOfMesureAppuisFlottants aSOMAF;
        for
                (
                 std::list<cSaisiePointeIm>::const_iterator itSP = mSOSPI.SaisiePointeIm().begin();
                 itSP != mSOSPI.SaisiePointeIm().end();
                 itSP++
                 )
        {
            cMesureAppuiFlottant1Im aMAF;
            aMAF.NameIm() = itSP->NameIm();

            for
                    (
                     std::list<cOneSaisie>::const_iterator itS=itSP->OneSaisie().begin();
                     itS!=itSP->OneSaisie().end();
                     itS++
                     )
            {
                if (itS->Etat()==eEPI_Valide)
                {
                    cOneMesureAF1I aM;
                    aM.NamePt() = itS->NamePt();
                    aM.PtIm() = itS->PtIm();
                    aMAF.OneMesureAF1I().push_back(aM);
                }
            }

            aSOMAF.MesureAppuiFlottant1Im().push_back(aMAF);
        }
        std::string aNameExp = DC()+StdPrefix(mParam.ExportPointeImage().Val());

        MakeFileXML(aSOMAF, aNameExp + "-S2D.xml");


        cDicoAppuisFlottant aDico;
        for (std::list<cPointGlob>::const_iterator itP=mSPG.PointGlob().begin(); itP!=mSPG.PointGlob().end(); itP++)
        {
            if (itP->Mes3DExportable().ValWithDef(false) && itP->P3D().IsInit())
            {
                cOneAppuisDAF anAP;
                anAP.Pt() = itP->P3D().Val();
                anAP.NamePt() = itP->Name();
                anAP.Incertitude() = Pt3dr(1,1,1);

                aDico.OneAppuisDAF().push_back(anAP);
            }
        }

        MakeFileXML(aDico, aNameExp + "-S3D.xml");

        /*
*/
    }
    /*
  a voir si pb de versions sous commit
<<<<<<< .mine
    <DicoAppuisFlottant>
          <OneAppuisDAF>
               <Pt>  103 -645 5</Pt>
               <NamePt>Coin-Gauche </NamePt>
               <Incertitude>  10 10 10  </Incertitude>
          </OneAppuisDAF>

=======
     A FUSIONNER AVEC LA VERSION SUR PC IGN, pas commite ???
     if (mParam.ExportPointeTerrain().IsInit())
     {
        cDicoAppuisFlottant aDic;
        for
        (
            std::list<cPointGlob>::iterator itP=mSPG.PointGlob().begin();
            itP!=mSPG.PointGlob().end();
            itP++
        )
        {
            if (itP->Mes3DExportable().ValWithDef(false))
            {
               cOneAppuisDAF anAp;
               anAp.Pt() = itP->P3D.Val();
               anAp.NamePt() = itP->Name();
               anAp.Incertitude() = Pt3dr(1,1,1);
               aDic.OneAppuisDAF().push_back(anAp);
            }
        }
        MakeFileXML(aDic, DC()+(mParam.ExportPointeTerrain().Val()));
     }
>>>>>>> .r889
*/
}


void  cAppli_SaisiePts::SetImagesVis(std::vector <cImage *> aImgs) 
{
   // std::cout << " cAppli_SaisiePts::SetImagesVis ### " << aImgs.size() << " " << mImagesVis.size() << "\n";
   mImagesVis = aImgs;
}

void cAppli_SaisiePts::Exit()
{
    Save();
    exit(-1);
}

double cAppli_SaisiePts::StatePriority(eEtatPointeImage aState)
{
    switch(aState)
    {
    case   eEPI_NonSaisi :
        return 1e3;
        break;

    case   eEPI_Refute :
        return mInterface->RefInvis() ? 0 : 1e-3;
        break;

    case   eEPI_Douteux :
        return 1;
        break;

    case eEPI_Valide :
        return 1e-6;
        break;

    case  eEPI_Disparu :
        return 0;
        break;

    case eEPI_NonValue :
    case eEPI_Highlight :
        break;
    }

    ELISE_ASSERT(false,"Unhandled Priority");
    return 0;
}

void   cAppli_SaisiePts::SetImagesPriority(cSP_PointGlob * PointPrio,bool aUseCpt)
{
    for (int aKI=0 ; aKI<int(mImagesTot.size()); aKI++)
    {
        cImage & anIm = *(mImagesTot[aKI]);
        anIm.SetPrio(anIm.CalcPriority(PointPrio,aUseCpt));
    }
}

void cAppli_SaisiePts::SortImages(std::vector<cImage *> &images)
{
/*
std::cout << "SOOiiiII " << images.size() << "\n";
for (int aK=0 ; aK<int(images.size()) ; aK++)
{
    std::cout << "iiiKKkk " << images[aK] << "\n";
}
*/
    cCmpIm aCmpIm(mInterface);
    std::sort(images.begin(),images.end(),aCmpIm);
}

void cAppli_SaisiePts::OnModifLoadedImage()
{
    for (int aK=0 ; aK<int(mImagesVis.size()) ; aK++)
    {
         mImagesVis[aK]->OnModifLoad();
    }
}

void cAppli_SaisiePts::ChangeImages
(
        cSP_PointGlob * PointPrio,
        const std::vector<cWinIm *>  &  aW2Ch,
        bool   aUseCpt
        )
{

    mImagesVis = mInterface->ComputeNewImagesPriority(PointPrio,aUseCpt);
/*
    SetImagesPriority(PointPrio,aUseCpt);
    SortImages(mImagesVis);
*/

#if (ELISE_X11)
    for (int aKW =0 ; aKW < int(aW2Ch.size()) ; aKW++)
    {
        aW2Ch[aKW]->SetNoImage();
    }
#endif
    int aKW =0;
    int aKI =0;

    while (aKW <int(aW2Ch.size()) )
    {
        ELISE_ASSERT(aKI<int(mImagesVis.size()),"Incoherence in cAppli_SaisiePts::ChangeImages");

        cImage * anIm = mImagesVis[aKI];

        if (!mInterface->isDisplayed(anIm))
        {
#if (ELISE_X11)
            aW2Ch[aKW]->SetNewImage(anIm);
#endif
            anIm->SetLoaded();
            aKW++;
        }
        aKI++;
    }
    OnModifLoadedImage();
}

bool cAppli_SaisiePts::HasOrientation() const
{
    return    mParam.KeyAssocOri().IsInit()
            && (mParam.KeyAssocOri().Val() != "NONE");
}




/*
*/


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
