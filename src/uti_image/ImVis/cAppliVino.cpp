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

#include "general/sys_dep.h"

#if (ELISE_X11)

#include "Vino.h"

std::string NameVinoPyramImage(const std::string & aDir,const std::string & aName,int aZoom)
{
     if (aZoom==1) return  aName;

     return  aDir + "Tmp-MM-Dir/" + StdPrefix(NameWithoutDir(aName)) +  "_Zoom" + ToString(aZoom) + ".tif";
}

std::string cAppli_Vino::NamePyramImage(int aZoom)
{
     return NameVinoPyramImage(mDir,mNameTiffIm,aZoom);
}

std::string  cAppli_Vino::CalculName(const std::string & aName,INT aZoom)
{
   return NameVinoPyramImage(mDir,aName,aZoom);
}

/*
*/


extern Video_Win * TheWinAffRed ;

// bool TreeMatchSpecif(const std::string & aNameFile,const std::string & aNameSpecif,const std::string & aNameObj);


cAppli_Vino::cAppli_Vino(int argc,char ** argv,const std::string & aNameImExtern,cAppli_Vino * aMother) :
    mSzIncr            (400,400),
    mNbPixMinFile      (2e6),
    mCurStats          (0),
    mTabulDynIsInit    (false),
    mNbHistoMax        (20000),
    mNbHisto           (mNbHistoMax),
    mHisto             (mNbHistoMax),
    mHistoLisse        (mNbHistoMax),
    mHistoCum          (mNbHistoMax),
    mIsMnt             (true),
    mWithBundlExp      (false),
    mClipIsChantier    (false),
    mMother            (aMother),
    mExtImNewP         ("Std"),
    mWithPCarac        (false),
    mSPC               (0),
    mQTPC              (0),
    mSeuilAC           (0.95),
    mSeuilContRel      (0.6),
    mCheckNuage        (nullptr),
    mCheckOri          (nullptr),
    mNameLab           ("eTPR_NoLabel"),
    mZoomCA            (10),

    //  Aime pts car
    mAimeShowFailed  (false),
    mWithAime         (false),
    mAimeSzW          (35,35),
    mAimeCW           (mAimeSzW / 2),
    mAimeZoomW        (7),
    mAimWStd          (nullptr),
    mAimWI0           (nullptr),
    mAimWLP           (nullptr)
{
    mNameXmlIn = Basic_XML_MM_File("Def_Xml_EnvVino.xml");
    if (argc>1)
    {
       mNameXmlOut =  DirOfFile(argv[1]) + "Tmp-MM-Dir/" + "EnvVino.xml";
       if (ELISE_fp::exist_file(mNameXmlOut))
       {
          if (TreeMatchSpecif(mNameXmlOut,"ParamChantierPhotogram.xml","Xml_EnvVino"))
              mNameXmlIn = mNameXmlOut;
       }
    }
    EnvXml() = StdGetFromPCP(mNameXmlIn,Xml_EnvVino);

    if (argc>1)
    {
        std::string aNameFile = NameWithoutDir(argv[1]);
        mStatIsInFile = false;
        std::list<cXml_StatVino> & aLStat = Stats();
        for (std::list<cXml_StatVino>::iterator itS=aLStat.begin(); itS!=aLStat.end() ; itS++)
        {
             if (itS->NameFile() == aNameFile)
             {
                 mStatIsInFile = true;
                 mCurStats = & (*itS);
             }
        }
        if (!mStatIsInFile)
        {
            aLStat.push_back(cXml_StatVino());
            mCurStats = & aLStat.back();
            mCurStats->Type() = eDynVinoModulo;
            mCurStats->IsInit() = false ;
            mCurStats->NameFile() = aNameFile;
            mCurStats->MulDyn() = 0.75;
        }
    }

    std::vector<std::string>  mParamClipCh;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mNameIm, "Image", eSAM_IsExistFile),
        LArgMain()  << EAM(SzW(),"SzW",true,"size of window")
                    << EAM(ZoomBilin(),"Bilin",true,"Bilinear mode")
                    << EAM(SpeedZoomGrab(),"SZG",true,"Speed Zoom Grab")
                    << EAM(SpeedZoomMolette(),"SZM",true,"Speed Zoom Molette")
                    << EAM(LargAsc(),"WS",true,"Width Scroller")
                    << EAM(mCurStats->IntervDyn(),"Dyn",true,"Max Min value for dynamic")
                    << EAM(ForceGray(),"Gray",true,"Force gray images (def=false)")
                    << EAM(mIsMnt,"IsMnt",true,"Display altitude if true, def exist of Mnt Meta data")
                    << EAM(mFileMnt,"FileMnt",true,"Default toto.tif -> toto.xml")
                    << EAM(mParamClipCh,"ClipCh",true,"Param 4 Clip Chantier [PatClip,OriClip]")
                    << EAM(mImNewP,"NewP",true,"Image for new tie point, if =\"\" curent image")
                    << EAM(mExtImNewP,"ExtImNewP",true,"Extension for new tie point, def=Std")
                    << EAM(mImSift,"ImSift",true,"Image for sift if != curent image")
                    << EAM(mSzSift,"ResolSift",true,"Resol of sift point to visualize")
                    << EAM(mPatSecIm,"PSI",true,"Pattern Imaage Second")
                    << EAM(mCheckHom,"CheckH",true,"Check Hom : [Cloud,Ori]")
                    << EAM(mNameLab,"Label",true,"Label for New Point ")
                    // << EAM(mCurStats->IntervDyn(),"Dyn",true,"Max Min value for dynamic")
         //  Aime pts car
                    << EAM(mNameAimePCar,"AimeNPC",true,"Aime name pts carac")
                    << EAM(mAimeShowFailed,"AimeSF",true,"Aime Show Failed points")
    );

    mLabel =   Str2eTypePtRemark(mNameLab);

    if (aNameImExtern !="")
      mNameIm = aNameImExtern;

// Files
    mDir = DirOfFile(mNameIm);
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    ELISE_fp::MkDirSvp(mDir+"Tmp-MM-Dir/");
    // MakeFileXML(EnvXml(),mNameXmlOut);


    mNameHisto = mDir + "Tmp-MM-Dir/Histo" + mNameIm + ".tif";
    if (ELISE_fp::exist_file(mNameHisto))
    {
        Tiff_Im aTH(mNameHisto.c_str());
        mNbHisto = aTH.sz().x;
        mHistoCum.Resize(mNbHisto);
        ELISE_COPY(aTH.all_pts(),aTH.in(), mHistoCum.out().chc(FX));
    }

    mNameIm = NameWithoutDir(mNameIm);
    mTiffIm = new Tiff_Im(Tiff_Im::StdConvGen(mDir+mNameIm,(ForceGray()?1:-1),true,false));
    mNameTiffIm = mTiffIm->name();
    mTifSz = mTiffIm->sz();
    mNbPix = double(mTifSz.x) * double(mTifSz.y);
    mNbChan = mTiffIm->nb_chan();
    mCoul = ( mNbChan == 3) ;

    mSzEl = (mNbChan  * mTiffIm->bitpp()) /8.0;

    if (!mStatIsInFile)
    {
        if (! EAMIsInit(&(mCurStats->IntervDyn()  ))) 
            mCurStats->IntervDyn() = Pt2dr(0,255);
    }
    if (EAMIsInit(&(mCurStats->IntervDyn())))
    {
          mCurStats->Type() = eDynVinoMaxMin;
    }
    SaveState();

//  MNT
   mFileMnt = StdPrefix(mNameIm) + ".xml";
   mFOM     = OptStdGetFromPCP(mFileMnt,FileOriMnt);
   if ((mNbChan!=1) || (mFOM==0))
       mIsMnt = false;

// window

    mRatioFulXY = Pt2dr(SzW()).dcbyc(Pt2dr(mTifSz));
    mRatioFul = ElMin(mRatioFulXY.x,mRatioFulXY.y);
    // mSzW = round_up(Pt2dr(mTifSz) * mRatioFul);
    Pt2di aSzW(SzW().x,LargAsc());
    if (aMother==0)
       mWAscH = Video_Win::PtrWStd(aSzW,true);
    else
    {
        mWAscH = new Video_Win(aMother->mWAscH->disp(),aMother->mWAscH->sop(),Pt2di(0,0),aSzW);
    }
    mW =    new  Video_Win(*mWAscH,Video_Win::eBasG,SzW());
    mWAscV = new Video_Win(*mW,Video_Win::eDroiteH,Pt2di(LargAsc(),SzW().y));


    mTitle = std::string("MicMac/Vino -> ") + mNameIm;
    // mDisp =  aMother ?  aMother->mDisp : new Video_Display(mW->disp());
    mDisp =   new Video_Display(mW->disp());



    
// Scrollers

    // mVVE = new  VideoWin_Visu_ElImScr(*mW,(true?Elise_Palette(mW->prgb()):Elise_Palette(mW->pgray())),mSzIncr);
    mVVE = new  VideoWin_Visu_ElImScr(*mW,(mCoul?Elise_Palette(mW->prgb()):Elise_Palette(mW->pgray())),mSzIncr);

    mVEch.push_back(1); 
    int aSc = 1;
    double aSzLim = SzLimSsEch().Val();
    while ( ((mNbPix*mSzEl)/ElSquare(aSc))  > aSzLim)
    {
        aSzLim = 6e6; // Si on reduit, tant qu'a faire on 
        aSc *= 2;
        mVEch.push_back(aSc); 
        std::string aName = NamePyramImage(mVEch.back());

        if (! ELISE_fp::exist_file(aName))
        {
             std::string aMes = "Reduce : " +ToString(aSc);
             mW->fixed_string(Pt2dr(100,SzW().y-50),aMes.c_str() ,mWAscH->prgb()(255,0,0),true);
             TheWinAffRed = mW;
             MakeTiffRed2
             (
                 NamePyramImage(mVEch.back()/2),
                 aName,
                 mTiffIm->type_el(),
                 16,
                 false,
                 -1
             );
             TheWinAffRed = 0;
             mW->clear();
        }
    }
    mWAscH->clear();

    Tiff_Im aTifHelp = MMIcone("Help");
    mWHelp = new Video_Win(*mW,Video_Win::eSamePos,aTifHelp.sz());
    ELISE_COPY(mWHelp->all_pts(),aTifHelp.in(),mWHelp->ogray());
    mWHelp->move_to(Pt2di(mW->sz().x-mWHelp->sz().x,LargAsc()+10));

    InitMenu();
    mW->set_title(mTitle.c_str());

    if (IsMatriceExportBundle(mNameIm))
    {
        std::string aNameSens = mDir+"Sensib-Data.dmp";
        if (ELISE_fp::exist_file(aNameSens))
        {
           mBundlExp = StdGetFromAp(aNameSens,XmlNameSensibs);
           mWithBundlExp = true;
        }
    }

    if (EAMIsInit(&mParamClipCh))
    {
       ELISE_ASSERT(mParamClipCh.size()>=2,"Not Enough arg in ParamClipCh");
       ELISE_ASSERT(mParamClipCh.size()<=2,"Too much Enough arg in ParamClipCh");

       mClipIsChantier = true;
       mPatClipCh = mParamClipCh[0];
       mOriClipCh = mParamClipCh[1];
    }

    if (EAMIsInit(&mImNewP))
    {
        if (mImNewP=="")
           mImNewP = mNameIm;
        mWithPCarac = true;

        // ELISE_ASSERT(false,"Vino revoir labels");
        mSPC =  LoadStdSetCarac(mLabel,mImNewP,mExtImNewP);
        mQTPC = new tQtOPC (mArgQt,Box2dr(Pt2dr(-10,-10),Pt2dr(10,10)+Pt2dr(mTifSz)),5,euclid(mTifSz)/50.0);

        for (int aKP=0 ; aKP<int(mSPC->OnePCarac().size()) ; aKP++)
        {
            mSPC->OnePCarac()[aKP].Id() = aKP;
            mQTPC->insert(&(mSPC->OnePCarac()[aKP]));
        }

    }
    if (EAMIsInit(&mSzSift))
    {
        if (EAMIsInit(&mImSift))
        {
        }
        else if (EAMIsInit(&mImNewP))
        {
            mImSift = mImNewP;
        }
        else
        {
            mImSift = mNameIm;
        }
        mWithPCarac = true;
        getPastisGrayscaleFilename(mDir,mImSift,mSzSift,mNameSift);
        mNameSift  = DirOfFile(mNameSift) + "LBPp" + NameWithoutDir(mNameSift) + ".dat";
        if (mSzSift<0) mNameSift = "Pastis/" + mNameSift;

        Tiff_Im aFileInit = PastisTif(mImSift);
        Pt2di       imageSize = aFileInit.sz();

        mSSF =  (mSzSift<0) ? 1.0 :   double( ElMax( imageSize.x, imageSize.y ) ) / double( mSzSift ) ;


        // std::cout << "NAMEPAST=" << mNameSift << "\n";
        // getchar();
        bool Ok = read_siftPoint_list(mNameSift,mVSift);
        if (!Ok)
        {
           std::cout << "Name sift=[" << mNameSift << "]\n";
           ELISE_ASSERT(Ok,"Bad read sift file\n");
        }
        // std::cout << "SIIIIffrt " << Ok << " Nb=" << mVSift.size() << "\n";
    }

    if (EAMIsInit(&mPatSecIm) && (aNameImExtern==""))
    {
       cElemAppliSetFile anEASF(mPatSecIm);
       const cInterfChantierNameManipulateur::tSet *  aVIS = anEASF.SetIm();
       for (const auto & aNIS : *aVIS)
       {
           mAVSI.push_back(new cAppli_Vino(argc,argv,aNIS,this));
       }
    }

    if (EAMIsInit(&mCheckHom))
    {
        ELISE_ASSERT(mCheckHom.size()==2,"cAppli_Vino, size CheckHom");
        // Cas master
        if (! mMother)
        {
            mCheckNuage =   cElNuage3DMaille::FromFileIm(mCheckHom[0]);
        }
        else
        {
            StdCorrecNameOrient(mCheckHom[1],mDir);
            mCheckOri = mICNM->StdCamGenerikOfNames(mCheckHom[1],mNameIm);
        }
    }
    if (EAMIsInit(&mNameAimePCar))
    {
       mWithAime = true;
       mWithPCarac = true;

       mDirAime =  "./Tmp-2007-Dir-PCar/"+ mNameIm + "/";
       std::string aPat =  "STD-V1AimePCar-" + mNameAimePCar + "-Tile0_0.dmp";
       cInterfChantierNameManipulateur* aAimeICNM = cInterfChantierNameManipulateur::BasicAlloc(mDirAime);

       cSetName * aSN= aAimeICNM->KeyOrPatSelector(aPat);

       if (aSN->Get()->empty())
       {
           std::cout << "DIR=" << mDirAime << "\n";
           std::cout << "PAT=" << aPat << "\n";
           ELISE_ASSERT(false,"No file for Aime visualization");
       }
       for (const auto & aName : *(aSN->Get()))
       {
           std::cout << "AIMEPCAR NAME=["<< aName << "]\n";
           mAimePCar.push_back(StdGetFromNRPH(mDirAime+aName,Xml2007SetPtOneType));
       }
       std::cout << "\n";


       // Fenetre pour voir l'image initiale en zoom
       mAimWI0  = mW->PtrChc(Pt2dr(0,0),Pt2dr(mAimeZoomW,mAimeZoomW));
       // Fenetre pour voir l'image initiale  de caracteristique
       mAimWStd  = mW->PtrChc(Pt2dr(-(2+mAimeSzW.x),0),Pt2dr(mAimeZoomW,mAimeZoomW));

       double aTr =  4+ 2 * mAimeSzW.x;
       double aZ  = 18.0;
       mAimWLP   = mW->PtrChc(Pt2dr(-1-aTr *(mAimeZoomW/aZ),-1.8),Pt2dr(aZ,aZ));
    }
}

void cAppli_Vino::PostInitVirtual()
{
    // mVVE->SetEtalDyn(mCurStats->IntervDyn().x,mCurStats->IntervDyn().y);
    mVVE->SetChgDyn(this);
    mScr = ElPyramScroller::StdPyramide(*mVVE,mNameTiffIm,&mVEch,false,false,this);
    mScr->SetAlwaysQuick(false);
    //mScr->SetAlwaysQuickInZoom(!ZoomBilin());
    SetInterpoleMode(ZoomBilin() ? eInterpolBiLin : eInterpolPPV,false);
    // mScr = new ImFileScroller<U_INT1> (*mVVE,*mTiffIm,1.0);
    // mScr->ReInitTifFile(*mTiffIm);
    InitTabulDyn();
    mScr->set_max();
    ShowAsc();

    for (auto  & aPtrAp : mAVSI)
    {
        aPtrAp->PostInitVirtual();
    }

    // Calcul des homologues par nuage

    if ((!mMother) && mSPC && mCheckNuage && (mAVSI.size()==1))
    {
        ElTimer aT0;
        std::cout << "BEGIN NEAREST \n";
        cBasicGeomCap3D * aCap2 = mAVSI[0]->mCheckOri;
        int aNbH=0;
        for (const auto & aPt :  mSPC->OnePCarac())
        {
            const cOnePCarac * aHom = nullptr;
            Pt2dr aPIm = mCheckNuage->Plani2Index(aPt.Pt());
            if (mCheckNuage->CaptHasData(aPIm))
            {
                Pt3dr aPTer = mCheckNuage->PtOfIndexInterpol(aPIm);
                if (aCap2->PIsVisibleInImage(aPTer))
                {
                    Pt2dr aPIm2 = aCap2->Ter2Capteur(aPTer);
                    double aDist;
                    double aSeuilD=2.0;
                    aHom = mAVSI[0]->Nearest(aSeuilD,aPIm2,&aDist,aPt.Kind());
                    if (aDist>aSeuilD)
                    {
                        ELISE_ASSERT(aHom==nullptr,"Incoherence in Nearest");
                        // aHom = nullptr;
                    }
                }
            }
            if (aHom) 
               aNbH++;
            mVptHom.push_back(aHom);
        }
        std::cout << "% Homol got " << (aNbH*100.0) / mVptHom.size()  << " T=" << aT0.uval() << "\n";

    }
}


void cAppli_Vino::SaveState()
{
    MakeFileXML(EnvXml(),mNameXmlOut);
}

void cAppli_Vino::Boucle()
{
   while (1)
   {
      Clik aCl = mDisp->clik_press();
      ExeOneClik(aCl);
      for (auto  & aPtrAp : mAVSI)
      {
         aPtrAp->ExeOneClik(aCl);
      }
   }
}


void cAppli_Vino::ExeOneClik(Clik & aCl)
{
   mP0Click =  aCl._pt;
   mScale0  =  mScr->sc();
   mTr0     =  mScr->tr();
   mBut0    = aCl._b;
   mCtrl0  = aCl.controled();
   mShift0  = aCl.shifted();

   // Click sur la fenetre principale 
   if (aCl._w == *mW)
   {
      if (mBut0==2)
         ExeClikGeom(aCl);
      if ((mBut0==4) || (mBut0==5))
      {
         ZoomMolette();
         ShowAsc();
      }

      if (mBut0==1)
      {
         if (mSPC && mCtrl0)
         {
             ShowSPC(mP0Click);
         }
         else if (mWithAime && mCtrl0)
         {
             InspectAime(mP0Click);
         }
         else
            GrabShowOneVal();
      }
      if (mBut0==3)
      {
         MenuPopUp();
      }
   }
   if (aCl._w == *mWAscH)
   {
      mModeGrab= eModeGrapAscX;
      mWAscH->grab(*this);
      ShowAsc();
   }
   if (aCl._w == *mWAscV)
   {
      mModeGrab= eModeGrapAscY;
      mWAscV->grab(*this);
      ShowAsc();
   }
   if (aCl._w == *mWHelp)
   {
      Help();
   }
}


void cAppli_Vino::Help()
{
    mW->fill_rect(Pt2dr(0,0),Pt2dr(mW->sz()),mW->pdisc()(P8COL::white));
    
    PutFileText(*mW,Basic_XML_MM_File("HelpVino.txt"));
    mW->clik_in();
    Refresh();
}

void  cAppli_Vino::EditData()
{
    cElXMLTree *  aTree = ToXMLTree(EnvXml());
    cWXXVinoSelector aSelector(mNameIm);
    cElXMLTree aFilter(Basic_XML_MM_File("FilterVino.xml"));

    cWindowXmlEditor aWX(*mW,true,aTree,&aSelector,&aFilter);

    aWX.TopDraw();
    aWX.Interact();

    cXml_EnvVino aNewEnv;
    xml_init(aNewEnv,aTree);
    EnvXml() = aNewEnv;

    cWXXInfoCase * aCaseID = aWX.GetCaseOfNam("IntervDyn",false);
    cWXXInfoCase * aCaseMD = aWX.GetCaseOfNam("MulDyn",false);

    if (aCaseID->mTimeModif >=0)
    {
        mCurStats->Type() = eDynVinoMaxMin;
    }

    if (aCaseMD->mTimeModif> ElMax(-1,aCaseID->mTimeModif))
    {
          mCurStats->Type() = eDynVinoStat2;
    }


    InitTabulDyn();


    Refresh();
}


/********************************************/
/*                                          */
/*  Lecture                                 */
/*                                          */
/********************************************/



cXml_StatVino  cAppli_Vino::StatRect(Pt2di  &aP0,Pt2di & aP1)
{
    cXml_StatVino aRes;
    CorrectRect(aP0,aP1,mTifSz);

    FillStat(aRes,rectangle(aP0,aP1),mTiffIm->in());

    return aRes;

}

void  cAppli_Vino::GrabShowOneVal()
{
    mModeGrab = eModeGrapShowRadiom;
    mW->grab(*this);
    EffaceMessageVal();
}



void  cAppli_Vino::ShowOneVal(Pt2dr aPW)
{
    Pt2di  aP = round_ni(mScr->to_user(aPW));
    Pt2di  aPp1 = aP+Pt2di(1,1);

    cXml_StatVino aStat = StatRect(aP,aPp1);

    //std::cout << "PPPPP " << aP << " " << mSom[0] << "\n";

    std::string aMesXY = " x=" + ToString(aP.x) + " y=" + ToString(aP.y);

    if (mWithBundlExp)
    {
         cSensibDateOneInc aSX = mBundlExp.SensibDateOneInc()[aP.x];
         cSensibDateOneInc aSY = mBundlExp.SensibDateOneInc()[aP.y];
         aMesXY =    "[" + aSX.NameBloc()  + ":" + aSX.NameInc()  + "] " 
                   + "[" +  aSY.NameBloc()  + ":" + aSY.NameInc() + "]"
                   +  "  (P=" + ToString(aP.x) + ","  + ToString(aP.y) +")"  ;
    }
    std::string aMesV =  " V=";
    for (int aK=0 ; aK<mNbChan; aK++)
    {
        if (aK!=0) aMesV = aMesV + " ";
        double aVal = aStat.Soms()[aK];
        if (mIsMnt)
        {
           aVal = ToMnt(*mFOM,aVal);
        }
        aMesV = aMesV  + StrNbChifApresVirg(aVal,3) ;
    }
        // aMesV = aMesV  + StrNbChifSign(aStat.Soms()[aK],3) + " ";
        // aMesV = aMesV  + SimplString(ToString(aStat.Soms()[aK])) + " ";

    EffaceMessageVal();
    //  On my computer with small pixel/big curset I dont see the messag
    int OffsetY = (MPD_MM()) ? -50 : -50;

    mVBoxMessageVal.push_back(PutMessage(aPW+Pt2dr(-20,OffsetY),aMesXY,P8COL::black));

    mVBoxMessageVal.push_back(PutMessage(Pt2dr(mVBoxMessageVal.back()._p0)+Pt2dr(0,-5),aMesV,P8COL::black));

/*
    Box2di = mVBoxMessageVal.P0(
Box2di cAppli_Vino::PutMessage(Pt2dr aP0 ,const std::string & aMes,int aCoulText,Pt2dr aSzRelief,int aCoulRelief)
    mP0StrVal = Pt2di(aPW)+Pt2di(-20,30);
    mW->fixed_string(Pt2dr(mP0StrVal),aMesV.c_str(),mW->pdisc()(P8COL::black),true);
    Pt2di aSzV =  mW->SizeFixedString(aMesV);
    Pt2di aSzXY =  mW->SizeFixedString(aMesXY);

    Pt2di aP0XY = mP0StrVal + Pt2di(0,aSzXY.y+3);
    mP0StrVal.y -= aSzV.y;
    mW->fixed_string(Pt2dr(aP0XY),aMesXY.c_str(),mW->pdisc()(P8COL::black),true);


    mP1StrVal = aP0XY + Pt2di(ElMax(aSzV.x,aSzXY.x),0);
    mInitP0StrVal = true;
*/
}


#endif



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
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
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
