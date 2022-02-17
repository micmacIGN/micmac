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

#include "CompColore.h"


/*************************************************/
/*                                               */
/*          cCC_Appli                            */
/*                                               */
/*************************************************/





cCC_Appli::cCC_Appli     
(
    const cCreateCompColoree & aCC,
    int argc,
    char ** argv
) :
    mWorkDir  (StdWokdDir(aCC.WorkDir(),argv[1])),
    mComInit  (ToCommande(argc,argv)),
    mCCC      (aCC),
    mICNM     (cInterfChantierNameManipulateur::StdAlloc
                    (
                         argc,argv,
		         WorkDir(),
			 mCCC.FileChantierNameDescripteur(),
			 mCCC.DicoLoc().PtrVal()
		    )
              ),
    mScF      (mCCC.ScaleFus().Val()),
    mMaxChOut (-1),
    mModeMaping   ( IsActive(aCC.MapCCC())),
    mInOutInit (false)
{
}

void cCC_Appli::DoStat()
{
   for 
   (
       std::list<cShowCalibsRel>::const_iterator itSCR=mCCC.ShowCalibsRel().begin();
       itSCR!=mCCC.ShowCalibsRel().end();
       itSCR++
   )
   {
      const std::vector<int> & aVC = itSCR->Channel();
      for (int aK1=0 ; aK1<int(aVC.size()) ; aK1++)
      {
         for (int aK2=aK1+1 ; aK2<int(aVC.size()) ; aK2++)
         {
             int aKC1= aVC[aK1];
             int aKC2= aVC[aK2];
             ELISE_ASSERT
             ( 
                   (aKC1>=0)  && (aKC1<int(mChOut.size())) 
                && (aKC2>=0)  && (aKC2<int(mChOut.size())),
                "Bad channel number in cCC_Appli::DoStat"
             );
             double aSigma;
             double aR =  mChOut[aKC1]->CalibRel(*mChOut[aKC2],itSCR->MaxRatio().Val(),aSigma);
             std::cout << aKC1 << " " << aKC2 << " Ratio " << aR <<  " Ecart " << aSigma << "\n";
         }
      }
      std::cout << "\n";
   }
}



void cCC_Appli::DoCalc()
{
    mMaster   = new cCC_OneChannel(mScF,mCCC.ImMaitresse(),*this);
    // Initialisation des images secondaires + virtual init de ts le monde
    mMaster->VirtualPostInit();
    mVAllsInit.push_back(mMaster);
    for 
    (
       std::list<cImSec>::iterator itS=mCCC.ImSec().begin();
       itS!=mCCC.ImSec().end();
       itS++
    )
    {
        mVSecInit.push_back(new cCC_OneChannelSec(mScF,*itS,*mMaster,*this));
        mVSecInit.back()->VirtualPostInit();
        mVAllsInit.push_back(mVSecInit.back());
    }

    // Calcul de la boite de  l'espace de fusion et de calcul

    if (mCCC.EnglobImMaitre().IsInit())
    {
        mBoxEspaceFus = R2ISup(mMaster->GlobBoxFus());
    }
    else if (mCCC.EnglobAll().IsInit())
    {
        mBoxEspaceFus = R2ISup(mMaster->GlobBoxFus());
        for (int aK=0 ; aK<int(mVSecInit.size()) ; aK++)
            mBoxEspaceFus = Sup(mBoxEspaceFus,R2ISup(mVSecInit[aK]->GlobBoxFus()));
    }
    else if (mCCC.EnglobBoxMaitresse().IsInit())
    {
        mBoxEspaceFus = R2ISup(mMaster->BoxIm2BoxFus(I2R(mCCC.EnglobBoxMaitresse().Val())));
    }

    std::cout << "Box Espace Fusion " << mBoxEspaceFus._p0 << mBoxEspaceFus._p1 << "\n";

    mBoxEspaceCalc = mCCC.BoxCalc().ValWithDef(mBoxEspaceFus);

    // Verif / affichage precision
    for (int aK=0 ; aK<int(mVSecInit.size()) ; aK++)
    {
	mVSecInit[aK]->TestVerif();
    }


    mNameMasterTiffIn = mMaster->NameCorrige();

   int aRab = 0;

   for 
   (
      std::list<cResultCompCol>::const_iterator itRCC = mCCC.ResultCompCol().begin();
      itRCC != mCCC.ResultCompCol().end();
      itRCC++
   )
   {
         if (itRCC->ImResultCC().ImResultCC_Cnes().IsInit())
         {
               const cImResultCC_Cnes & aCnes = itRCC->ImResultCC().ImResultCC_Cnes().Val();
               if (aCnes.SzIterFCSte().IsInit())
               {
                   aRab = ElMax(aRab,aCnes.SzIterFCSte().Val()*aCnes.NbIterFCSte().Val());
               }
         }
   }

   aRab += aRabOutput;

  //=====================   cCC_Appli::DoCalc ==   
  //=====================   cCC_Appli::DoCalc ==   
  //=====================   cCC_Appli::DoCalc ==   
  //=====================   cCC_Appli::DoCalc ==   
 

  if (mCCC.TailleBloc().IsInit())
  {
   // On genere des process si ByProcess et qu'on n'est pas justement en train de se
   // faire rappeler par process (ie si KBoxParal est init)
      cEl_GPAO * aGPAO = (mCCC.ByProcess().IsInit() && (! mCCC.KBoxParal().IsInit()))  ?
                         new cEl_GPAO :
                         0;

      cElTask *  aGlobTask = aGPAO ? &(aGPAO->GetOrCreate("all","")) : 0 ;



      int aTB=mCCC.TailleBloc().Val();
      Pt2di aPtBrd(aRab,aRab);
      Box2di aBoxBrd(-aPtBrd,aPtBrd);
      cDecoupageInterv2D aDI2(mBoxEspaceCalc,Pt2di(aTB,aTB),aBoxBrd);

      for (mKBox=0 ; mKBox<aDI2.NbInterv() ; mKBox++)
      {
          bool OK=true;
          
          
          if (mCCC.KBoxParal().IsInit())
          {
              OK= (mKBox== mCCC.KBoxParal().Val());
          }
          // La premiere dalle  la lance classique pour etre sur que les fichiers soient
          // crees une fois pour toutes
          if (aGPAO)
              OK = (mKBox==0);


          if (OK)
          {
             std::cout << "Reste " << aDI2.NbInterv() -mKBox << " Box\n";
             DoOneBoxOut(aDI2.KthIntervIn(mKBox),aDI2.KthIntervOut(mKBox));
          }

          if (aGPAO && (! OK))
          {
              std::string aName = std::string("Bloc") + ToString(mKBox);
              std::string aCom = mComInit + std::string(" KBoxParal=")+ ToString(mKBox);
              cElTask & aTK = aGPAO->NewTask(aName,aCom);
              aGlobTask->AddDep(aTK);
          }
      }
      if (aGPAO)
      {
         std::string aNameMk = "MakeSup";
         aGPAO->GenerateMakeFile(aNameMk);
         std::string aCom =   std::string("make all -f ") 
                            + aNameMk 
                            + std::string(" -j") + ToString(mCCC.ByProcess().Val());
         System(aCom);
      }
  }
  else
  {
      mKBox = 0;
      DoOneBoxOut(mBoxEspaceCalc,mBoxEspaceCalc);
  }

  DoStat();
}

bool  cCC_Appli::ModeMaping() const
{
   return mModeMaping;
}

void  cCC_Appli::DoMapping(int argc,char ** argv)
{
   mICNM->SetMapCmp(mCCC.MapCCC(),argc,argv);
}

cCC_Appli::~cCC_Appli()
{
    DeleteAndClear(mVSecInit);
    DeleteAndClear(mChOut);
}



//   ACCESSOR  
const cCreateCompColoree & cCC_Appli::CCC() { return mCCC; }
const std::string & cCC_Appli::WorkDir() { return mWorkDir; }
cInterfChantierNameManipulateur * cCC_Appli::ICNM() { return mICNM; }
cCC_OneChannel&  cCC_Appli::Master() { return  *mMaster; }


//   UTILITAIRES 
cChanelOut &  cCC_Appli::KCanOut(int aK)
{
   if ((aK<0) || (aK>=int(mChOut.size())))
   {
        std::cout << "For canal Out =" << aK << "\n";
        ELISE_ASSERT(false,"Bad Num cannal in cCC_Appli::KCanOut");
   }

   return *(mChOut[aK]);
}

Fonc_Num cCC_Appli::KF(int aK)
{
   return KCanOut(aK).Im()->in();
}

//  CALCUL

void cCC_Appli::DoOneBoxOut(const Box2di & aBoxOut,const Box2di & aBoxSauvOut)
{
    ElTimer aChrono;
    bool  ShowTime=true;

    bool aOkM = mMaster->SetBoxOut(aBoxOut);
    ELISE_ASSERT(aOkM,"Master Vide !!");

    mCurSecs.clear();
    mCurAllCh.clear();
    mCurAllCh.push_back(mMaster);
     

    for (int aK=0 ; aK<int(mVSecInit.size()) ; aK++)
    {
        cCC_OneChannelSec * aKCh = mVSecInit[aK];
	bool aOkSec = aKCh->SetBoxOut(aBoxOut);
        if (aOkSec)
        {
           mCurAllCh.push_back(aKCh);
           mCurSecs.push_back(aKCh);
        }
    }
    mNbChAll = mCurAllCh.size();

    if (ShowTime)
    {
      std::cout << "Time Init " << aChrono.uval() << "\n";
      aChrono.reinit();
    }

    CreateOrUpdateInOut(aBoxOut);
    if (ShowTime)
    {
      std::cout << "Time Lecture " << aChrono.uval() << "\n";
      aChrono.reinit();
    }


    InitByInterp();
    if (ShowTime)
    {
      std::cout << "Time Interp " << aChrono.uval() << "\n";
      aChrono.reinit();
    }

    SauvImg(aBoxOut,aBoxSauvOut);
    if (ShowTime)
    {
      std::cout << "Time Sauv " << aChrono.uval() << "\n";
      aChrono.reinit();
    }
}



void cCC_Appli::CreateOrUpdateInOut(const Box2di & aBoxOut)
{

   for (int aK=0 ; aK<mNbChAll ; aK++)
   {
       mCurAllCh[aK]->DoChIn();
       ElSetMax(mMaxChOut,mCurAllCh[aK]->MaxChOut());
   }

   mNbChOut = mMaxChOut+1;

   for (int aK=0 ; aK<mNbChOut ; aK++)
   {
       if (aK>=int(mChOut.size()))
          mChOut.push_back(new cChanelOut(aBoxOut.sz()));
       mChOut[aK]->ResizeAndReset(aBoxOut.sz());
   }

   for (int aKc=0 ; aKc<mNbChAll ; aKc++)
   {
       const  std::vector<cChannelIn *>  aVCI = mCurAllCh[aKc]->ChIn();
       for (int  aKi=0 ; aKi <int(aVCI.size()) ; aKi++)
       {
           int aKo = aVCI[aKi]->CCC().Out();
	   ELISE_ASSERT((aKo>=0) && (aKo<mNbChOut),"Incoherence in cCC_Appli::InitChInOut");
	   mChOut[aKo]->AddChIn(aVCI[aKi]);
       }
   }
/*
   for (int aK=0 ; aK<mNbChOut ; aK++)
   {
       ELISE_ASSERT
       (
          mChOut[aK]->ChIns().size()!=0,
	  "Ch Out empty"
        );
   }
*/
   mInOutInit = true;
}


void cCC_Appli::InitByInterp()
{
   for (int aK=0 ; aK<mNbChOut ; aK++)
   {
       mChOut[aK]->InitByInterp();
   }
}




void cCC_Appli::SauvImg(const Box2di & aBoxOut,const Box2di & aBoxSauvOut)
{
   for
   (
      std::list<cResultCompCol>::const_iterator itR=mCCC.ResultCompCol().begin();
      itR!=mCCC.ResultCompCol().end();
      itR++
   )
   {
       SauvImg(aBoxOut,aBoxSauvOut,*itR);
   }
}

std::vector<int> StdBF(const cTplValGesInit<std::vector<int> > & aV)
{
     if (aV.IsInit())
        return aV.Val();

     std::vector<int> aRes;
     for (int aK=0 ; aK<3 ; aK++)
          aRes.push_back(aK);

     return aRes;
}

Fonc_Num ThomSom(Fonc_Num aF,Fonc_Num aF0,const cImResultCC_Thom & aRT)
{
   bool OKSps = false;
   for (int aK=0 ; aK<aRT.NbIterPond().Val() ; aK++)
   {
       if (aRT.PondExp().IsInit())
       {
          aF = canny_exp_filt(aF,aRT.PondExp().Val(),aRT.PondExp().Val());
           OKSps = (aK==0);
       }
       else if (aRT.PondCste().IsInit())
       {
          aF = rect_som(aF,aRT.PondCste().Val());
          OKSps = (aK==0);
       }
   }

   if (aRT.SupressCentre().Val())
   {
         ELISE_ASSERT(OKSps,"Cannot supress centre in ThomSom");
         aF = aF - aF0;
   }

   return aF;
}

void cCC_Appli::SauvImg(const Box2di & aBoxOut,const Box2di & aBoxSauvOut,const cResultCompCol & aRCC)
{
   std::string aName=WorkDir()+mICNM->Assoc1To1(aRCC.KeyName(),mNameMasterTiffIn,true);


   Tiff_Im::PH_INTER_TYPE aPhT = Tiff_Im::BlackIsZero;
   Fonc_Num aFoncIn;


   if (aRCC.ImResultCC_Gray().IsInit())
   {
        const cImResultCC_Gray & aIG = aRCC.ImResultCC_Gray().Val();
        aFoncIn = KCanOut(aIG.Channel().Val()).Im()->in();
        aPhT = Tiff_Im::BlackIsZero;
   }
   else if (aRCC.ImResultCC_RVB().IsInit())
   {
        const cImResultCC_RVB & aIrgb = aRCC.ImResultCC_RVB().Val();
        Pt3di aAxes = aIrgb.Channel().Val();
        aFoncIn =Virgule
                 (
                     KCanOut(aAxes.x).Im()->in(),
                     KCanOut(aAxes.y).Im()->in(),
                     KCanOut(aAxes.z).Im()->in()
                 );
        aPhT = Tiff_Im::RGB;
   }
   else if (aRCC.ImResultCC_Cnes().IsInit())
   {
      const cImResultCC_Cnes  & aICn = aRCC.ImResultCC_Cnes().Val();

      //   Le signal haute frequence 
                // Masque
      cChanelOut &  aCHf  = KCanOut(aICn.ChannelHF().Val());
      Fonc_Num aFHF = aCHf.Im()->in_proj();
      Fonc_Num aFMOY;
      if (aICn.SzIterFCSte().IsInit())
      {
            aFMOY = aFHF;
            int aSzF = aICn.SzIterFCSte().Val();
            for (int aK=0 ; aK<aICn.NbIterFCSte().Val() ; aK++)
            {
                 if (aICn.ModeMedian().Val())
                     aFMOY = MedianBySort(aFHF,aSzF);
                  else
                     aFMOY = rect_som(aFHF,aSzF)/ElSquare(1.0+2*aSzF);
            }
      }
      // La c'est un mode ou on utilise une ponderation avec masque
      else
      {
         Pt2di aSzM = aICn.SzF().Val();
         Im2D_REAL8  aImF(aSzM.x,aSzM.y,aICn.ValueF().Val().c_str());

         double aSom=0;
         ELISE_COPY(aImF.all_pts(),aImF.in(),sigma(aSom));
         ELISE_COPY(aImF.all_pts(),aImF.in()/aSom,aImF.out());


         aFMOY = som_masq(Rconv(aFHF),aImF);
      }
      Symb_FNum aFPdr(aFHF/Max(1.0,aFMOY));


      //   Le signal basse frequence 
      std::vector<int> aChBF = StdBF(aICn.ChannelBF());

      //   Le melange

      for (int aK=0 ; aK<int(aChBF.size()) ; aK++)
      {
           Fonc_Num aFK =  KF(aChBF[aK])*aFPdr;
           if (aK==0)
              aFoncIn = aFK ;
           else 
              aFoncIn = Virgule(aFoncIn,aFK);
      }
      switch(aChBF.size())
      {
         case 1:
           aPhT = Tiff_Im::BlackIsZero;
         break;

         case 3:
           aPhT = Tiff_Im::RGB;
         break;

         default :
            ELISE_ASSERT(false,"Cannot determine photometrique interpretation");
         break;
      }
   }
   else if (aRCC.ImResultCC_PXs().IsInit())
   {
      aPhT = Tiff_Im::RGB;
      const cImResultCC_PXs aPxs = aRCC.ImResultCC_PXs().Val();
      int aKr=0,aKv=1,aKb=2,aKp=3;
      if (aPxs.Channel().IsInit())
      {
          std::vector<int> aVI = aPxs.Channel().Val();
          ELISE_ASSERT(aVI.size()==4,"ImResultCC_PXs.Channel() = Bad Size");
          aKr = aVI[0];
          aKv = aVI[1];
          aKb = aVI[2];
          aKp = aVI[3];
      }
      double aC = aPxs.Cste().Val();
      Pt3dr aAx = aPxs.AxeRGB().Val();
/*  On resoud l'equation en D :
    Gr = -C + a (R+aD) + b (V+bD) + c (B + c B)
    Car Gr = - C + aR + b V + c B est l'observation (eventuellement apprise)
    Et la contribution, D, de gris doit etre repartie proportionnellement a
   la correlation.
    Donc
       D = (Gr+C - aR - b V - c B ) / (a2+b2+c2)
*/
      if (aPxs.ApprentisageAxeRGB().Val())
      {
            int  Use[4] = {1,1,1,1};
            for
            (
                std::list<std::string>::const_iterator itUU=aPxs.UnusedAppr().begin();
                itUU!=aPxs.UnusedAppr().end();
                itUU++
            )
            {
               if (*itUU=="C")
                  Use[0] = 0;
               else if (*itUU=="R")
                  Use[1] = 0;
               else if (*itUU=="V")
                  Use[2] = 0;
               else if (*itUU=="B")
                  Use[3] = 0;
               else
               {
                  ELISE_ASSERT(false,"Bad key for UnusedAppr");
               }
            }
            L2SysSurResol  aSys(4);
            cChanelOut &  aCR = KCanOut(aKr);
            cChanelOut &  aCV = KCanOut(aKv);
            cChanelOut &  aCB = KCanOut(aKb);
            cChanelOut &  aCP = KCanOut(aKp);

            std::vector<int> aVInd;
            for (int aK=0; aK<4 ; aK++)
                aVInd.push_back(aK);
            if (0)
            {
               for (int aK=0 ; aK<4 ; aK++)
               {
                   if (Use[aK] == 0)
                   {
                       double aCoef[4] = {0,0,0,0};
                       aCoef[aK] = 1;
                       aSys.GSSR_AddNewEquation(1e3,aCoef,0,(double *)NULL);
                   }
               }
            }
            else
            {
                double aCoe1[4] = {0,1,-1,0};
                double aCoe2[4] = {0,1,0,-1};
 
                aSys.GSSR_AddContrainteIndexee(aVInd,aCoe1,0);
                aSys.GSSR_AddContrainteIndexee(aVInd,aCoe2,0);
            }


            int aStep = 10;
            Pt2di aP;
            Pt2di  aSzRas = aBoxOut.sz();
            for (aP.x=0 ; aP.x < aSzRas.x ; aP.x+=aStep)
            {
                for (aP.y=0 ; aP.y < aSzRas.y; aP.y+=aStep)
                {
                     double aCoef[4];
                     aCoef[0] = -1 * Use[0];
                     aCoef[1] = aCR.GetVal(aP) * Use[1];
                     aCoef[2] = aCV.GetVal(aP) * Use[2];
                     aCoef[3] = aCB.GetVal(aP) * Use[3];
                     aSys.GSSR_AddNewEquation(1.0,aCoef,aCP.GetVal(aP),(double *)NULL);
                }
            }


            Im1D_REAL8  aSol = aSys.GSSR_Solve(0);
            aC =aSol.data()[0];
            aAx = Pt3dr(aSol.data()[1],aSol.data()[2],aSol.data()[3]);

             std::cout << "Cste = " << aSol.data()[0] << "\n";
             std::cout << "R  = " << aSol.data()[1] << "\n";
             std::cout << "V  = " << aSol.data()[2] << "\n";
             std::cout << "B  = " << aSol.data()[3] << "\n";
      }
      Symb_FNum aFR =  KF(aKr);
      Symb_FNum aFV =  KF(aKv);
      Symb_FNum aFB =  KF(aKb);

      Symb_FNum aDeltaGr((KF(aKp) +aC  -aAx.x*aFR -aAx.y*aFV- aAx.z*aFB) / (aAx.x+aAx.y+aAx.z));
      aFoncIn =Virgule(
                        aFR + aDeltaGr,
                        aFV + aDeltaGr,
                        aFB + aDeltaGr
                 );
   }
   else if (aRCC.ImResultCC_Thom().IsInit())
   {
       aPhT = Tiff_Im::RGB;
       const cImResultCC_Thom & aRT = aRCC.ImResultCC_Thom().Val();
       std::vector<int> aChBF = StdBF(aRT.ChannelBF());

       cChanelOut &  aCHf  = KCanOut(aRT.ChannelHF().Val());
       Symb_FNum aFHF (aCHf.Im()->in_proj());
       Symb_FNum aFHF2 (aCHf.Im()->in_proj());
 
        Symb_FNum aSomP (  Rconv(ThomSom
                          (
                               Virgule(1,aFHF,Square(aFHF)),
                               Virgule(1,aFHF2,Square(aFHF2)),
                               aRT
                          )));
        Symb_FNum aS1 (aSomP.v0());
        Symb_FNum aSp (aSomP.v1());
        Symb_FNum aSpp (aSomP.v2());
        Symb_FNum aSp2(aSpp-Square(aSp)/aS1);

        Symb_FNum aSDet(aS1*aSpp-Square(aSp));

        for (int aK=0 ; aK<int(aChBF.size()) ; aK++)
        {
             cChanelOut &  aCHC  = KCanOut(aChBF[aK]);
             Symb_FNum   aFC(aCHC.Im()->in_proj());
             Symb_FNum   aFC2(aCHC.Im()->in_proj());

              Symb_FNum aSomCP (  Rconv(ThomSom
                                (
                                     Virgule(aFC,aFC*aCHf.Im()->in_proj()),
                                     Virgule(aFC2,aFC2*aCHf.Im()->in_proj()),
                                     aRT
                                )));

              Symb_FNum aSc = aSomCP.v0();
              Symb_FNum aScp = aSomCP.v1();

              Fonc_Num aSol= 0;

              if (aRT.MPDBidouille().IsInit())
              {
                  Fonc_Num aFDet =  Max(aRT.EcartMin(),aSDet);
                  Fonc_Num aSa =   (aSpp*aSc -aSp*aScp)/aFDet;
                  Fonc_Num aSb =   (-aSp*aSc+aS1*aScp)/aFDet;

                  aSol = aSa + aFHF2 * aSb;
              }
              else if (aRT.ThomBidouille().IsInit())
              {
                  double aKr = aRT.ThomBidouille().Val().PourCent() / 100.0;
                  double aMin = aRT.ThomBidouille().Val().VMin();
                  Fonc_Num  aScpCentre = aScp - (aSc*aSp) / aS1;
                  aSol = (
                             aSc +aKr* (aFHF2*aS1-aSp) * (aMin*aS1+aScpCentre)/ (aMin*aS1+aSp2)
                          ) /aS1;
                    // double
              }

              aFoncIn = (aK==0) ? aSol : Virgule(aFoncIn,aSol);

           
        }
   }

   
   GenIm::type_el aType = Xml2EL(aRCC.Type().Val());
   bool  isModif;
   Tiff_Im  aFile = Tiff_Im::CreateIfNeeded
            (
                isModif,
                aName,
                mBoxEspaceFus.sz(),
                aType,
                Tiff_Im::No_Compr,
                aPhT,
                  Tiff_Im::Empty_ARG
               +  Arg_Tiff(Tiff_Im::ANoStrip())
               +  Arg_Tiff(Tiff_Im::AFileTiling(Pt2di(20000,20000)))
            );

   if (aRCC.GamaExport().IsInit()||aRCC.LutExport().IsInit())
   {
      int aVMIn,aVMax;
      min_max_type_num(aType,aVMIn,aVMax);
      Im1D_INT4 aLut(1);

      double aCoeff = 10.0;

      if (aRCC.LutExport().IsInit())
      {
           aLut = LutIm(aRCC.LutExport().Val(),aVMIn,aVMax-1,aCoeff);
      }
      else if (aRCC.GamaExport().IsInit())
      {
           aCoeff=100.0;
           aLut = LutGama
                  (
                      1<<16,
                      aRCC.GamaExport().Val(),
                      aRCC.RefGama().Val(),
                      aVMax-1,
                      aCoeff
                  );
      }
      aFoncIn = SafeUseLut(aLut,aFoncIn,aCoeff);

   }
   else
   {
      if (type_im_integral(aType))
         aFoncIn = round_ni(aFoncIn);
      aFoncIn = Tronque(aType,aFoncIn);
   }


   Pt2di aP0 = aBoxSauvOut._p0- mBoxEspaceFus._p0;
   Pt2di aP1 = aP0 + aBoxSauvOut.sz();

   Pt2di aDec = aBoxOut._p0 - mBoxEspaceFus._p0;

// std::cout << "P0 P1 DEC " << aP0 << aP1 << aDec << "\n";
// std::cout << "Fux P0 " << mBoxEspaceFus._p0 << "\n";
// std::cout << "BixOut P0 " << aBoxOut._p0 << "\n";
// std::cout << "SauvOut P0 " << aBoxSauvOut._p0 << "\n";

   ELISE_COPY
   (
      rectangle(aP0,aP1),
      trans(aFoncIn,-aDec),
      aFile.out()
   );
}




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
