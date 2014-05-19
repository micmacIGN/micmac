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

// List  of classes
class cMCI_Appli;
// Contains the information to store each image : Radiometry & Geometry
class cMCI_Ima;

// classes declaration

class cMCI_Ima
{
    public:
       cMCI_Ima(cMCI_Appli & anAppli,const std::string & aName);

       Pt2dr ClikIn();
       // Renvoie le saut de prof pour avoir un pixel
       double EstimateStep(cMCI_Ima *);

       void  DrawFaisceaucReproj(cMCI_Ima & aMas,const Pt2dr & aP);
       Video_Win *  W() {return mW;};
       void InitMemImOrtho(cMCI_Ima *);

       void CalculImOrthoOfProf(double aProf,cMCI_Ima * aMaster);

       Fonc_Num  FCorrel(cMCI_Ima *);
       Pt2di Sz(){return mSz;}

    private :
       cMCI_Appli &   mAppli;
       std::string     mName;
       Tiff_Im         mTifIm;
       Pt2di           mSz;
       Im2D_U_INT1     mIm;
       Im2D_U_INT1     mImOrtho;


       Video_Win *     mW;
       std::string     mNameOri;
       CamStenope *    mCam;

};



class cMCI_Appli
{
    public :

        cMCI_Appli(int argc, char** argv);
        const std::string & Dir() const {return mDir;}
        bool ShowArgs() const {return mShowArgs;}
        std::string NameIm2NameOri(const std::string &) const;
        cInterfChantierNameManipulateur * ICNM() const {return mICNM;}

        Pt2dr ClikInMaster();

        void TestProj();
        void InitGeom();
        void AddEchInv(double aInvProf,double aStep)
        {
            mNbEchInv++;
            mMoyInvProf += aInvProf;
            mStep1Pix   += aStep;
        }
    private :
        cMCI_Appli(const cMCI_Appli &); // To avoid unwanted copies

        void DoShowArgs();



        std::string mFullName;
        std::string mDir;
        std::string mPat;
        std::string mOri;
        std::string mNameMast;
        std::list<std::string> mLFile;
        cInterfChantierNameManipulateur * mICNM;
        std::vector<cMCI_Ima *>          mIms;
        cMCI_Ima *                       mMastIm;
        bool                              mShowArgs;
        int                               mNbEchInv;
        double                            mMoyInvProf;
        double                            mStep1Pix;
};

/********************************************************************/
/*                                                                  */
/*         cMCI_Ima                                                 */
/*                                                                  */
/****************StdCorrecNameOrient****************************************************/

cMCI_Ima::cMCI_Ima(cMCI_Appli & anAppli,const std::string & aName) :
   mAppli  (anAppli),
   mName   (aName),
   mTifIm  (Tiff_Im::StdConvGen(mAppli.Dir() + mName,1,true)),
   mSz     (mTifIm.sz()),
   mIm     (mSz.x,mSz.y),
   mImOrtho (1,1),
   mW      (0),
   mNameOri (mAppli.NameIm2NameOri(mName)),
   mCam      (CamOrientGenFromFile(mNameOri,mAppli.ICNM()))
{
   ELISE_COPY(mIm.all_pts(),mTifIm.in(),mIm.out());

   if (0) // (mAppli.ShowArgs())
   {
       std::cout << mName << mSz << "\n";
       mW = Video_Win::PtrWStd(Pt2di(1200,800));
       mW->set_title(mName.c_str());
       ELISE_COPY(mW->all_pts(),mTifIm.in(),mW->ogray());
       //mW->clik_in();

       ELISE_COPY(mW->all_pts(),255-mIm.in(),mW->ogray());
       //mW->clik_in();
       std::cout << mNameOri
                 << " F=" << mCam->Focale()
                 << " P=" << mCam->GetProfondeur()
                 << " A=" << mCam->GetAltiSol()
                 << "\n";
          // 1- Test at very low level
        Im2D<U_INT1,INT> aImAlias = mIm;
        ELISE_ASSERT(aImAlias.data()==mIm.data(),"Data");
        U_INT1 ** aData = aImAlias.data();
        for (int anY=0 ; anY<mSz.y/3 ; anY++)
            for (int anX=0 ; anX<anY ; anX++)
            {
                 ElSwap(aData[anY][anX],aData[anX][anY]);
            }
        U_INT1 * aDataL = aImAlias.data_lin();

        for (int anY=0 ; anY<mSz.y ; anY++)
        {
            ELISE_ASSERT
            (
                aData[anY]==(aDataL+anY*mSz.x),
                "data"
            );
         }
         memset(aDataL+mSz.x*50,128,mSz.x*100);
         ELISE_COPY(mW->all_pts(),mIm.in(),mW->ogray());

         // 2- Test with functional approach

         ELISE_COPY
         (
            mIm.all_pts(),
            mTifIm.in(),
            // Output is directed both in window & Im
            mIm.out() | mW->ogray()
          );

          ELISE_COPY
         (
            disc(Pt2dr(200,200),150),
            255-mIm.in()[Virgule(FY,FX)],
            // Output is directed both in window & Im
             mW->ogray()
          );

          int aSzF = 20;
         ELISE_COPY
         (
            rectangle(Pt2di(0,0),Pt2di(400,500)),
            // rect_som(mIm.in(),20)/ElSquare(1+2*aSzF),
            rect_som(mIm.in_proj(),aSzF)/ElSquare(1+2*aSzF),
            // Output is directed both in window & Im
             mW->ogray()
          );

          Fonc_Num aF = mIm.in_proj();
          aSzF=4;
          int aNbIter = 5;
          for (int aK=0 ; aK<aNbIter ; aK++)
              aF = rect_som(aF,aSzF)/ElSquare(1+2*aSzF);

         ELISE_COPY(mIm.all_pts(),aF,mW->ogray());

        //  ELISE_COPY(mIm.all_pts(),mTifIm.in(),mIm.out());


         ELISE_COPY(mIm.all_pts(),mIm.in(),mW->ogray());

         // 3- Test with Tpl  approach

         Im2D<U_INT1,INT4> aDup(mSz.x,mSz.y);
         TIm2D<U_INT1,INT4> aTplDup(aDup);
         TIm2D<U_INT1,INT4> aTIm(mIm);

         for (int aK=0 ; aK<aNbIter ; aK++)
         {
            for (int anY=0 ; anY<mSz.y ; anY++)
                for (int anX=0 ; anX<mSz.x ; anX++)
                {
                    int aNbVois = ElSquare(1+2*aSzF);
                    int aSom=0;
                    for (int aDx=-aSzF; aDx<=aSzF ; aDx++)
                       for (int aDy=-aSzF; aDy<=aSzF ; aDy++)
                           aSom += aTIm.getproj(Pt2di(anX+aDx,anY+aDy));
                    aTplDup.oset(Pt2di(anX,anY),aSom/aNbVois);
                }

            Pt2di aP;
            for (aP.y=0 ; aP.y<mSz.y ; aP.y++)
                for (aP.x=0 ; aP.x<mSz.x ; aP.x++)
                    aTIm.oset(aP,aTplDup.get(aP));
         }
         ELISE_COPY(mIm.all_pts(),mIm.in(),mW->ogray());
   }



}

void cMCI_Ima::CalculImOrthoOfProf(double aProf,cMCI_Ima * aMaster)
{
    TIm2D<U_INT1,INT> aTIm(mIm);
    TIm2D<U_INT1,INT> aTImOrtho(mImOrtho);
    int aSsEch = 10;
    Pt2di aSzR = aMaster->mSz/ aSsEch;
    TIm2D<float,double> aImX(aSzR);
    TIm2D<float,double> aImY(aSzR);

    Pt2di aP;
    for (aP.x=0 ; aP.x<aSzR.x; aP.x++)
    {
        for (aP.y=0 ; aP.y<aSzR.y; aP.y++)
        {
            Pt3dr aPTer = aMaster->mCam->ImEtProf2Terrain(Pt2dr(aP*aSsEch),aProf);
            Pt2dr aPIm = mCam->R3toF2(aPTer);
            aImX.oset(aP,aPIm.x);
            aImY.oset(aP,aPIm.y);
        }
    }

    for (aP.x=0 ; aP.x<aMaster->mSz.x; aP.x++)
    {
        for (aP.y=0 ; aP.y<aMaster->mSz.y; aP.y++)
        {
            /*
            Pt3dr aPTer = aMaster->mCam->ImEtProf2Terrain(Pt2dr(aP),aProf);
            Pt2dr aPIm0 = mCam->R3toF2(aPTer);
            */
            Pt2dr aPInt = Pt2dr(aP) / double(aSsEch);
            Pt2dr aPIm (aImX.getr(aPInt,0),aImY.getr(aPInt,0));
            float aVal = aTIm.getr(aPIm,0);
            aTImOrtho.oset(aP,round_ni(aVal));
        }
    }

    if ( 0 &&  (mName=="Abbey-IMG_0250.jpg"))
    {
        static Video_Win * aW = Video_Win::PtrWStd(Pt2di(1200,800));
        ELISE_COPY(mImOrtho.all_pts(),mImOrtho.in(),aW->ogray());
    }
}


Fonc_Num  cMCI_Ima::FCorrel(cMCI_Ima *aMaster)
{
    int aSzW = 2;
    double aNbW = ElSquare(1+2*aSzW);
    Fonc_Num aF1 = mImOrtho.in_proj();
    Fonc_Num aF2 = aMaster->mImOrtho.in_proj();


    Fonc_Num aS1 = rect_som(aF1,aSzW) / aNbW;
    Fonc_Num aS2 = rect_som(aF2,aSzW) / aNbW;

    Fonc_Num aS12 = rect_som(aF1*aF2,aSzW) / aNbW - aS1*aS2;
    Fonc_Num aS11 = rect_som(Square(aF1),aSzW) / aNbW - Square(aS1);
    Fonc_Num aS22 = rect_som(Square(aF2),aSzW) / aNbW - Square(aS2);

    Fonc_Num aRes = aS12 / sqrt(Max(1e-5,aS11*aS22));

//static Video_Win * aW = Video_Win::PtrWStd(Pt2di(1200,800));
//ELISE_COPY(aW->all_pts(),128*(1+aRes),aW->ogray());

    return aRes;
}


void cMCI_Ima::InitMemImOrtho(cMCI_Ima * aMas)
{
    mImOrtho.Resize(aMas->mIm.sz());
}

Pt2dr cMCI_Ima::ClikIn()
{
    return mW->clik_in()._pt;
}

void  cMCI_Ima::DrawFaisceaucReproj(cMCI_Ima & aMas,const Pt2dr & aP)
{
    if (! mW) return ;
    double aProfMoy =  aMas.mCam->GetProfondeur();
    double aCoef = 1.2;

    std::vector<Pt2dr> aVProj;
    for (double aMul = 0.2; aMul < 5; aMul *=aCoef)
    {
         Pt3dr aP3d =  aMas.mCam->ImEtProf2Terrain(aP,aProfMoy*aMul);
         Pt2dr aPIm = this->mCam->R3toF2(aP3d);

         aVProj.push_back(aPIm);
    }
    for (int aK=0 ; aK<((int) aVProj.size()-1) ; aK++)
        mW->draw_seg(aVProj[aK],aVProj[aK+1],mW->pdisc()(P8COL::red));
}

double cMCI_Ima::EstimateStep(cMCI_Ima * aMas)
{

   std::string aKey = "NKS-Assoc-CplIm2Hom@@dat";

   std::string aNameH =   mAppli.Dir()
                        + mAppli.ICNM()->Assoc1To2
                        (
                            aKey,
                            this->mName,
                            aMas->mName,
                            true
                        );
   ElPackHomologue aPack = ElPackHomologue::FromFile(aNameH);

   Pt3dr aDirK = aMas->mCam->DirK();
   for
   (
       ElPackHomologue::iterator iTH = aPack.begin();
       iTH != aPack.end();
       iTH++
   )
   {
       Pt2dr aPInit1 = iTH->P1();
       Pt2dr aPInit2 = iTH->P2();

       double aDist;
       Pt3dr aTer = mCam->PseudoInter(aPInit1,*(aMas->mCam),aPInit2,&aDist);

       double aProf2 = aMas->mCam->ProfInDir(aTer,aDirK);


       Pt2dr aProj1 = mCam->R3toF2(aTer);
       Pt2dr aProj2 = aMas->mCam->R3toF2(aTer);

      // std::cout << aMas->mCam->ImEtProf2Terrain(aProj2,aProf2) -aTer << "\n";


       if (0)
          std::cout << "Ter " << aDist << " " << aProf2
                 << " Pix " << euclid(aPInit1,aProj1)
                 << " Pix " << euclid(aPInit2,aProj2) << "\n";
       double aDeltaProf = aProf2 * 0.0002343;
       Pt3dr aTerPert = aMas->mCam->ImEtProf2Terrain
                      (aProj2,aProf2+aDeltaProf);

       Pt2dr aProjPert1 = mCam->R3toF2(aTerPert);

       double aDelta1Pix = aDeltaProf / euclid(aProj1,aProjPert1);
       double aDeltaInv = aDelta1Pix / ElSquare(aProf2);
       // std::cout << "Firts Ecart " << aDelta1Pix << " "<< aDeltaInv  << "\n";
       mAppli.AddEchInv(1/aProf2,aDeltaInv);
   }
   return aPack.size();


}

/********************************************************************/
/*                                                                  */
/*         cMCI_Appli                                               */
/*                                                                  */
/********************************************************************/


cMCI_Appli::cMCI_Appli(int argc, char** argv):
    mNbEchInv (0),
    mMoyInvProf (0),
    mStep1Pix    (0)
{
    // Reading parameter : check and  convert strings to low level objects
    mShowArgs=false;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mFullName,"Full Name (Dir+Pat)")
                    << EAMC(mNameMast,"Name of Master Image")
                    << EAMC(mOri,"Used orientation"),
        LArgMain()  << EAM(mShowArgs,"Show",true,"Give details on args")
    );

    // Initialize name manipulator & files
    SplitDirAndFile(mDir,mPat,mFullName);
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    mLFile = mICNM->StdGetListOfFile(mPat);

    StdCorrecNameOrient(mOri,mDir);

    if (mShowArgs) DoShowArgs();

    // Initialize all the images structure
    mMastIm = 0;
    for (
              std::list<std::string>::iterator itS=mLFile.begin();
              itS!=mLFile.end();
              itS++
              )
     {
           cMCI_Ima * aNewIm = new  cMCI_Ima(*this,*itS);
           mIms.push_back(aNewIm);
           if (*itS==mNameMast)
               mMastIm = aNewIm;
     }

     // Ckeck the master is included in the pattern
     ELISE_ASSERT
     (
        mMastIm!=0,
        "Master image not found in pattern"
     );

     if (mShowArgs)
        TestProj();

     InitGeom();
     Pt2di aSz = mMastIm->Sz();
     Im2D_REAL4 aImCorrel(aSz.x,aSz.y);
     Im2D_REAL4 aImCorrelMax(aSz.x,aSz.y,-10);
     Im2D_INT2  aImPax(aSz.x,aSz.y);


     double aStep = 0.5;
     for (int aKPax = -60 ; aKPax <=60 ; aKPax++)
     {
         std::cout << "ORTHO at " << aKPax << "\n";
         double aInvProf = mMoyInvProf + aKPax * mStep1Pix * aStep;
         double aProf = 1/aInvProf;

         for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
              mIms[aKIm]->CalculImOrthoOfProf(aProf,mMastIm);

         Fonc_Num aFCorrel = 0;
         for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
         {
             cMCI_Ima * anIm = mIms[aKIm];
             if (anIm != mMastIm)
                 aFCorrel = aFCorrel+anIm->FCorrel(mMastIm);
         }
         ELISE_COPY(aImCorrel.all_pts(),aFCorrel,aImCorrel.out());
         ELISE_COPY
         (
            select(aImCorrel.all_pts(),aImCorrel.in()>aImCorrelMax.in()),
            Virgule(aImCorrel.in(),aKPax),
            Virgule(aImCorrelMax.out(),aImPax.out())
         );
     }
     Video_Win aW = Video_Win::WStd(Pt2di(1200,800),true);
     ELISE_COPY(aW.all_pts(),aImPax.in()*6,aW.ocirc());
     aW.clik_in();

}

void cMCI_Appli::InitGeom()
{
    for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
    {
        cMCI_Ima * anIm = mIms[aKIm];
        if (anIm != mMastIm)
        {
            anIm->EstimateStep(mMastIm) ;
        }
        anIm->InitMemImOrtho(mMastIm) ;
    }
    mMoyInvProf /= mNbEchInv;
    mStep1Pix /= mNbEchInv;
}

void cMCI_Appli::TestProj()
{
    if (! mMastIm->W()) return;
    while (1)
    {
        Pt2dr aP = ClikInMaster();
        for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
        {
            mIms[aKIm]->DrawFaisceaucReproj(*mMastIm,aP);
        }
    }
}


Pt2dr cMCI_Appli::ClikInMaster()
{
    return mMastIm->ClikIn();
}


std::string cMCI_Appli::NameIm2NameOri(const std::string & aNameIm) const
{
    return mICNM->Assoc1To1
    (
        "NKS-Assoc-Im2Orient@-"+mOri+"@",
        aNameIm,
        true
    );
}

void cMCI_Appli::DoShowArgs()
{
     std::cout << "DIR=" << mDir << " Pat=" << mPat << " Orient=" << mOri<< "\n";
     std::cout << "Nb Files " << mLFile.size() << "\n";
     for (
              std::list<std::string>::iterator itS=mLFile.begin();
              itS!=mLFile.end();
              itS++
              )
      {
              std::cout << "    F=" << *itS << "\n";
      }
}

/********************************************************************/
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/

int ExoMCI_main(int argc, char** argv)
{
   cMCI_Appli anAppli(argc,argv);

   return EXIT_SUCCESS;
}


int ExoMCI_2_main(int argc, char** argv)
{
   std::string aNameFile;
   double D=1.0;

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameFile,"Name of GCP file"),
        LArgMain()  << EAM(D,"D",true,"Unused")
    );

    cDicoAppuisFlottant aDico=  StdGetFromPCP(aNameFile,DicoAppuisFlottant);


    std::cout << "Nb Pts " <<  aDico.OneAppuisDAF().size() << "\n";
    std::list<cOneAppuisDAF> & aLGCP =  aDico.OneAppuisDAF();

    for (
             std::list<cOneAppuisDAF>::iterator iT= aLGCP.begin();
             iT != aLGCP.end();
             iT++
    )
    {
        // iT->Pt() equiv a (*iTp).Pt()
        std::cout << iT->NamePt() << " " << iT->Pt() << "\n";
    }

   return EXIT_SUCCESS;
}

int ExoMCI_1_main(int argc, char** argv)
{
   int I,J;
   double D=1.0;

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(I,"Left Operand")
                    << EAMC(J,"Right Operand"),
        LArgMain()  << EAM(D,"D",true,"divisor of I+J")
    );

    std::cout << "(I+J)/D = " <<  (I+J)/D << "\n";

   return EXIT_SUCCESS;
}


int ExoMCI_0_main(int argc, char** argv)
{

   for (int aK=0 ; aK<argc ; aK++)
      std::cout << " Argv[" << aK << "]=" << argv[aK] << "\n";

   return EXIT_SUCCESS;
}

int  ExoCorrelEpip_main(int argc,char ** argv)
{
    std::string aNameI1,aNameI2;
    int aPxMax= 199;
    int aSzW = 5;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameI1,"Name Image1")
                    << EAMC(aNameI2,"Name Image2"),
        LArgMain()  << EAM(aPxMax,"PxMax",true,"Pax Max")
    );

    Im2D_U_INT1 aI1 = Im2D_U_INT1::FromFileStd(aNameI1);
    Im2D_U_INT1 aI2 = Im2D_U_INT1::FromFileStd(aNameI2);

    Pt2di aSz1 = aI1.sz();
    Im2D_REAL4  aIScoreMin(aSz1.x,aSz1.y,1e10);
    Im2D_REAL4  aIScore(aSz1.x,aSz1.y);
    Im2D_INT2   aIPaxOpt(aSz1.x,aSz1.y);

    Video_Win aW = Video_Win::WStd(Pt2di(1200,800),true);

    for (int aPax = -aPxMax ; aPax <=aPxMax ; aPax++)
    {
        std::cout << "PAX tested " << aPax << "\n";
        Fonc_Num aI2Tr = trans(aI2.in_proj(),Pt2di(aPax,0));
        ELISE_COPY
        (
             aI1.all_pts(),
             rect_som(Abs(aI1.in_proj()-aI2Tr),aSzW),
             aIScore.out()
        );
        ELISE_COPY
        (
           select(aI1.all_pts(),aIScore.in()<aIScoreMin.in()),
           Virgule(aPax,aIScore.in()),
           Virgule(aIPaxOpt.out(),aIScoreMin.out())
        );
    }

    ELISE_COPY
    (
        aW.all_pts(),
        aIPaxOpt.in()[Virgule(FY,FX)]*3,
        aW.ocirc()
     );
     aW.clik_in();



    return EXIT_SUCCESS;

}

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
