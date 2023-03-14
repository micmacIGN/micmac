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

void saisieMasq_ElInitArgMain(int argc,char ** argv, std::string &aFullName, std::string &aPost, std::string &aNameMasq, std::string &aAttr, Pt2di &aSzW, double &aGama,bool & aForceTif)
{
    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aFullName,"Name of input image", eSAM_IsExistFile) ,
           LArgMain() << EAM(aSzW,"SzW",true)
                      << EAM(aPost,"Post",true)
                      << EAM(aNameMasq,"Name",true,"Name of result, default toto->toto_Masq.tif")
                      << EAM(aGama,"Gama",true)
                      << EAM(aAttr,"Attr",true)
                      << EAM(aForceTif,"ForceTif",true,"Force tif post (do not maintain tiff,TIF ...) def=true")
    );
}

#if (ELISE_X11)


class cAppliSM : public Grab_Untill_Realeased,
                 public cClikInterceptor
{
    public :
         cAppliSM(int argc,char ** argv,bool ForV2);

     void NoOp(){}
     void ExeClik(Clik aCl);
// std::cout << "FFFFF  "<< aNTifIm << "\n";

         void Help();

         void DumpImage(const std::string &);
         int  CurCoul() {return mCaseCoul->Val() ? 1 : 0;}

    private :
        void  GUR_query_pointer(Clik,bool);
        bool InterceptClik(Clik);
        void ShowCC();

        void GetPolyg(Clik aCl);
        void DoMenu(Clik aCl);

        Pt2di                  mSzWP;
        std::string            mDir;
        std::string            mNIm;

    Tiff_Im *           mT1;
    Video_Win*          mW;
    Video_Win*          mWCC;
        EliseStdImageInteractor * mESII;
        BiScroller *              mBiS;

        Tiff_Im                 mTifExit;
        Pt2di                   mSzCase;

        GridPopUpMenuTransp*    mPopUp;
        CaseGPUMT *             mCaseExit;
        BoolCaseGPUMT *         mCaseCoul;
        bool                    mEnd;
        bool                    mForceTif;
        std::string             mAttr;


        Video_Win  *           mWHelp;
};

void cAppliSM::GUR_query_pointer(Clik cl,bool)
{
    mPopUp->SetPtActif(Pt2di(cl._pt));
}

void cAppliSM::GetPolyg(Clik aCl)
{
   std::vector<Pt2dr> aVPt =
        mESII->GetPolyg(P8COL::red,P8COL::green,mESII->W2U(aCl._pt));
    if (int(aVPt.size()) > 2)
    {
       ElList<Pt2di> aLPt;

       for (int aK=0 ; aK<int(aVPt.size()) ; aK++)
           aLPt = aLPt + round_ni(aVPt[aK]);
       ELISE_COPY
       (
         polygone(aLPt),
         CurCoul(),
         mBiS-> ImMasq().out()
       );
    }
    mESII->Refresh();
}

void  cAppliSM::DumpImage(const std::string & aName)
{

/*
    Pt2di aSz = mW->sz();

    ElXim  anIm (*mW,aSz);

    Im2D_U_INT1 aIR(aSz.x,aSz.y);
    Im2D_U_INT1 aIG(aSz.x,aSz.y);
    Im2D_U_INT1 aIB(aSz.x,aSz.y);


   anIm.read_in_el_image(Pt2di(0,0),Pt2di(0,0),aSz,aIR,aIG,aIB);

   Tiff_Im::Create8BFromFonc
   (
        aName,
        aSz,
        Virgule(aIR.in(),aIG.in(),aIB.in())
   );
*/
   mW->DumpImage(aName);
   std::cout << "Dumped " << aName << "\n"; getchar();

}



void cAppliSM::ShowCC()
{
    Box2dr aBox(Pt2di(0,0),mWCC->sz());

    Fill_St aFst(mWCC->pdisc()(CurCoul()? P8COL::yellow:P8COL::white));
    mWCC->fill_rect(aBox._p0,aBox._p1,aFst);

    mWCC->draw_rect(aBox,Line_St(mWCC->pdisc()(P8COL::red),4));
}

void cAppliSM::DoMenu(Clik aCl)
{
   mPopUp->UpCenter(Pt2di(aCl._pt));
// DumpImage("toto.tif");
   mW->grab(*this);
   CaseGPUMT * aCase =  mPopUp->PopAndGet();

   if (aCase== mCaseExit)
   {
      mEnd=true;
   }
   else if (aCase==mCaseCoul)
   {
       ShowCC();
   }
   else
   {
   }
}


void PutFileText(Video_Win,const std::string &);

void cAppliSM::Help()
{
    mW->fill_rect(Pt2dr(0,0),Pt2dr(mW->sz()),mW->pdisc()(P8COL::white));

    PutFileText(*mW,Basic_XML_MM_File("HelpSaisieMasq.txt"));
    mW->clik_in();
    mESII->Refresh();
    // Refresh();

}


bool cAppliSM::InterceptClik(Clik aCl)
{
   if (aCl._w==*mWHelp) 
   {
       Help();
       return true;
   }

   if (aCl._w==*mWCC) 
   {
      mCaseCoul->SetVal(!mCaseCoul->Val());
      ShowCC();
      return true;
   }


   return false;

}


void cAppliSM::ExeClik(Clik aCl)
{
    std::cout  << mESII->W2U(aCl._pt)  << aCl._b << "\n";

    if (InterceptClik(aCl))
    {
        return;
    }
    else if (aCl._b==1)
    {
       GetPolyg(aCl);
    }
    else if (aCl._b==3)
    {
       DoMenu(aCl);
    }
}

std::string  DirMMVII_AndCreate
             (
	          const std::string & aType, 
	          const std::string & aValue 
	     )
{
    static std::string 	DirMMVI = "MMVII-PhgrProj";

    std::string aRes = DirMMVI + ELISE_STR_DIR + aType + ELISE_STR_DIR + aValue + ELISE_STR_DIR;
    ELISE_fp::MkDirRec(aRes);
	
    return aRes;
}



cAppliSM::cAppliSM(int argc,char ** argv,bool ForV2) :
   // mTifExit  ("data/Exit.tif"),
   mTifExit  (MMIcone("Exit")),
   mSzCase   (mTifExit.sz()),
   mEnd      (false),
   mForceTif (true)
{
    mSzWP   = Pt2di(900,700);
    std::string aFullName;
    std::string aPost("Masq");
    std::string aNameMasq ="";
    std::string aNTifIm;
    double aGama=1.0;

    if (ForV2)
    {
         std::string aDirV2 = "Std";
         ElInitArgMain
         (
                argc,argv,
                LArgMain() << EAMC(aFullName,"Name of input image", eSAM_IsExistFile) ,
                LArgMain() << EAM(mSzWP,"SzW",true)
                           << EAM(aGama,"Gama",true)
                           << EAM(aDirV2,"SubDir",true,"Sub Directory for MMVII, Defaut="+aDirV2)

         );

         aNTifIm = NameFileStd(aFullName,1,false);
	 aNameMasq = DirMMVII_AndCreate("Mask",aDirV2) + aFullName + ".tif";
    }
    else
    {
       saisieMasq_ElInitArgMain(argc, argv, aFullName, aPost, aNameMasq, mAttr, mSzWP, aGama,mForceTif);
       aPost = aPost + mAttr;

       SplitDirAndFile(mDir,mNIm,aFullName);
       std::string aTifPost;
       if (IsPostfixed(mNIm))
       {
          aTifPost = StdPostfix(mNIm);
       }

       if ((! IsKnownTifPost(aTifPost)) ||  mForceTif)
       {
         aTifPost = "tif";
       }
     
       aTifPost = "." + aTifPost;

       aNTifIm = NameFileStd(mDir+mNIm,1,false);

       if (aNameMasq=="")
          aNameMasq = mDir+StdPrefixGen(mNIm)+ "_" + aPost + aTifPost;
       else
       {
          // aNameMasq = mDir+aNameMasq ;
       }
    }

    if (!ELISE_fp::exist_file(aNameMasq))
    {
         Tiff_Im::SetDefTileFile(1<<20);
         Tiff_Im aFIm=Tiff_Im::StdConv(aNTifIm);
         Tiff_Im aFM
                 (   aNameMasq.c_str(),
                      aFIm.sz(),
                      GenIm::bits1_msbf,
                      Tiff_Im::No_Compr,
                      Tiff_Im::BlackIsZero
                 );

         ELISE_COPY(aFM.all_pts(),0,aFM.out());
    }

    mT1  = new Tiff_Im(Tiff_Im::StdConvGen(aNTifIm,1,false));


    if (!ForV2)
    {
        std::string aNameXML = StdPrefix(aNameMasq)+".xml";
        if (!ELISE_fp::exist_file(aNameXML))
        {
           cFileOriMnt anOri;

           anOri.NameFileMnt() = aNameMasq;
           anOri.NombrePixels() = mT1->sz();
           anOri.OriginePlani() = Pt2dr(0,0);
           anOri.ResolutionPlani() = Pt2dr(1.0,1.0);
           anOri.OrigineAlti() = 0.0;
           anOri.ResolutionAlti() = 1.0;
           anOri.Geometrie() = eGeomMNTFaisceauIm1PrCh_Px1D;

           MakeFileXML(anOri,aNameXML);
        }
        std::cout << aNameXML << "\n";
    }



    mW =  Video_Win::PtrWStd(mSzWP);
    mWCC = new Video_Win(Video_Win(*mW,Video_Win::eDroiteH,Pt2di(100,50)));

    Tiff_Im aTifHelp = MMIcone("Help");
    mWHelp = new Video_Win(*mWCC,Video_Win::eBasG,aTifHelp.sz());
    ELISE_COPY(mWHelp->all_pts(),aTifHelp.in(),mWHelp->ogray());

    VideoWin_Visu_ElImScr * aVVE = new VideoWin_Visu_ElImScr(*mW,*mT1);
    // ElPyramScroller * aPyr = ElImScroller::StdPyramide(*aVVE,aNTifIm);

// std::cout << "EEEEEEEEEEEE  "<< aNTifIm << "\n";
    mBiS =  BiScroller::MasqBiscroller(*aVVE,aNTifIm,aNameMasq);
// std::cout << "FFFFF  "<< aNTifIm << "\n";

   mBiS->GraySetGamaCorrec(aGama);

    mESII = new EliseStdImageInteractor(*mW,*mBiS,2,5,4,this);


    mPopUp = new GridPopUpMenuTransp(*mW,mSzCase,Pt2di(5,5),Pt2di(1,1));


    mCaseExit = new CaseGPUMT
                    (
                       *mPopUp,"titi",Pt2di(4,4),
                       MMIcone("Exit").in(0) *255
                    );

    mCaseCoul  = new BoolCaseGPUMT
                    (
                       *mPopUp,"titi",Pt2di(0,0),
                       MMIcone("Coul").in(0) *255,
                       (!MMIcone("Coul").in(0)) *255,
                       true
                    );

    ShowCC();

    while (! mEnd)
    {
         ExeClik(mESII->clik_press())  ;
    }


    Tiff_Im aTifM = Tiff_Im::StdConv(aNameMasq);
    int aMax,aMin;
    min_max_type_num(aTifM.type_el(),aMin,aMax);

//   std::cout  << "MM " << aMin << " " << aMax << "\n";
    ELISE_COPY
    (
       mBiS-> ImMasq().all_pts(),
       mBiS-> ImMasq().in() * (aMax-1),
       aTifM.out()
    );


    //mWHelp->move_to(Pt2di(0));

}

int MMVII_SaisieMasq_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);
    cAppliSM aAP(argc,argv,true);
    return EXIT_SUCCESS;
}

int SaisieMasq_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);
    cAppliSM aAP(argc,argv,false);
    return EXIT_SUCCESS;
}

#endif





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
