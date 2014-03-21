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
#include "../src/uti_phgrm/MICMAC/MICMAC.h"

double GenereRandom(const Pt2dr & aP,double aPow)
{
   double aR = NRrandom3();
   if (aPow !=1)
     aR *= pow(aR,aPow);
   return aP.x + (aP.y-aP.x)  * aR;
}

    //======================================================================
    //======================================================================
    //======================================================================

void cAppliMICMAC::SimulateRelief(bool isNew)
{
    if (! isNew)
    {
       if (! mSRel->DoItR().Val())
          return;
    }
    std::cout << "DOING  RELIEF\n";


    Pt2di aSz(mTifMnt->sz());
    Box2dr aBox(Pt2dr(0,0),Pt2dr(aSz));

    Im2D_REAL4 aMne(aSz.x,aSz.y,0.0);

    Fonc_Num aFoncMnt= 0.0;
    Pt2dr aPG = mSRel->PenteGlob().Val();
    aFoncMnt = aFoncMnt+ aPG.x*(FX-aSz.x/2.0) +  aPG.y*(FY-aSz.y/2.0);

    for 
    (
        std::list<cFoncPer>::const_iterator itF=mSRel->FoncPer().begin(); 
        itF!=mSRel->FoncPer().end() ; 
        itF++
    )
    {
        Pt2dr aPer = itF->Per();
        double aAmpl = itF->Ampl();
        if (itF->AmplIsDer().Val())
           aAmpl *= euclid(aPer)/(2*PI);
        aPer =  aPer/ mRRGlob;
        aPer  = aPer * ((2*PI) / (square_euclid(aPer)));

        aFoncMnt  = aFoncMnt + sin(FX*aPer.x + FY*aPer.y) * aAmpl; 
        
    }

    for 
    (
        std::list<cSimulBarres>::const_iterator itSB=mSRel->SimulBarres().begin();
        itSB!=mSRel->SimulBarres().end() ;
        itSB++

    )
    {
        const cSimulBarres & aSB = *itSB;
        for (int aKs=0 ; aKs<aSB.Nb() ; aKs++)
        {
            double aLon = GenereRandom(aSB.IntervLongeur(),aSB.PowDistLongueur().Val()) / mRRGlob;
            double aLar = GenereRandom(aSB.IntervLargeur(),1.0) / mRRGlob;
            double aHaut = GenereRandom(aSB.IntervHauteur(),1.0) ;
            double aPente = GenereRandom(aSB.IntervPentes(),1.0);

            Pt2dr aC = aBox.RandomlyGenereInside();
            Pt2dr aV = Pt2dr::FromPolar(1.0,2*PI*NRrandom3());

            Pt2dr aT = aV * (aLon+aLar+2);
            Pt2dr aN = aV * Pt2dr(0,1) * (aLar+2.0);
            
            ElList<Pt2di> aL;
            aL = aL+ round_ni(aC+aT +aN);
            aL = aL+ round_ni(aC+aT -aN);
            aL = aL+ round_ni(aC-aT -aN);
            aL = aL+ round_ni(aC-aT +aN);

            aT = aV;
            aN = aT * Pt2dr(0,1);
            Pt2dr aP0 = aC + aT*aLon;
            Pt2dr aP1 = aC - aT*aLon;

            Pt2d<Fonc_Num> aPC(FX-aC.x,FY-aC.y);
            Pt2d<Fonc_Num> aFN(aN.x,aN.y);
            Pt2d<Fonc_Num> aFT(aT.x,aT.y);
            Fonc_Num aDist = Abs(scal(aPC,aFN));

            aDist = Max(aDist,scal(aFT,Pt2d<Fonc_Num>(FX-aP0.x,FY-aP0.y)));
            aDist = Max(aDist,scal(-aFT,Pt2d<Fonc_Num>(FX-aP1.x,FY-aP1.y)));

            aDist = Max(0,aLar-aDist)*aPente;
            Symb_FNum aSD(aDist);
            if (aHaut>0)
            {
                aDist = aSD + aHaut*(aSD>0);
            }

            bool isSortant = NRrandom3() < aSB.ProbSortant().Val();
            if (!isSortant)
               aDist = - aDist;

            // aDist = Abs(scal(aFT,Pt2d<Fonc_Num>(FX-aP0.x,FY-aP0.y)));
           

   // std::cout << aLar << " " << aLon << " " << euclid(aP0,aC) << "\n";

            ELISE_COPY(polygone(aL),aDist,aMne.out());
            // Pt2dr aP1 = aC+aV +aT +aN;
            // Pt2dr aP2 = aC+aV +aT -aN;
            // Pt2dr aP2 = aC-aV;
        }
    }

    ELISE_COPY
    (
         aMne.all_pts(),
         (aMne.in() + aFoncMnt) * mSimFactPA,
         mTifMnt->out()
    );
    std::cout << "---- DONE RELIEF\n";
}
    //======================================================================
    //======================================================================
    //======================================================================

class cMMSimulZBuf :  public cStateSimul,
                      public cZBuffer
{
    public :
         cMMSimulZBuf(const cStateSimul &,const cPriseDeVue &);
    public :
         const cPriseDeVue& mPDV;
         const cGeomImage & mGeoI;
         Pt2di              mSzTer;
         Im2D_REAL4         mImT;
         TIm2D<REAL4,REAL8> mTImT;
         float **           mDataZ;
         Box2dr             mBoxIm;

         cTplCIKTabul<float,double>  * mInterpTabul;
         cInterpolateurIm2D<float>  * mInterp;
         //cTplCIKTabul<U_INT1,int>  * mInterpIm;
         cInterpolBicubique<U_INT1>  * mInterpIm;
         int                          mSzK;
         Box2di                       mBoxTerInterp;

         Pt3dr ProjTerrain(const Pt3dr & aP) const ;
         double ZofXY(const Pt2di & aP)   const  ; // En general le MNT
         bool SelectPBascul(const Pt2dr & aP)   const;
         bool SelectP(const Pt2di & aP)   const;

        virtual double ZInterpofXY(const Pt2dr & aP,bool & OK) const;
        virtual  Pt3dr InvProjTerrain(const Pt3dr &) const;

};


double cMMSimulZBuf::ZInterpofXY(const Pt2dr & aP,bool & OK) const
{
   
   OK = mBoxTerInterp.inside(round_ni(aP));
   if (OK)
   {
  // std::cout << ZofXY(aP) << " " << mInterp->GetVal(mDataZ,aP-mSimIntBxTer._p0) << "\n";
      return mInterp->GetVal(mDataZ,aP-Pt2dr(mSimIntBxTer._p0));
   }
   return 0;
}



double cMMSimulZBuf::ZofXY(const Pt2di & aP)   const  
{
   // std::cout << aP  << mImT.sz() << "\n";
   return mTImT.get(aP-mSimIntBxTer._p0);
} 

bool cMMSimulZBuf::SelectP(const Pt2di & aP)   const
{
   return mSimIntBxTer.inside(aP);
}


bool cMMSimulZBuf::SelectPBascul(const Pt2dr & aP)   const
{
   return mBoxIm.inside(aP);
}

Pt3dr cMMSimulZBuf::ProjTerrain(const Pt3dr & aP) const 
{
    
   Pt2dr aPIm = mGeoI.Objet2ImageInit_Euclid(Pt2dr(aP.x,aP.y),&aP.z);
   return Pt3dr(aPIm.x,aPIm.y,aP.z);
}

Pt3dr cMMSimulZBuf::InvProjTerrain(const Pt3dr & aP) const
{
   Pt2dr aPIm = mGeoI.ImageAndPx2Obj_Euclid(Pt2dr(aP.x,aP.y),&aP.z);
   return Pt3dr(aPIm.x,aPIm.y,aP.z);
}


cMMSimulZBuf::cMMSimulZBuf(const cStateSimul & aSM,const cPriseDeVue & aPDV) :
    cStateSimul (aSM),
    cZBuffer
    (
        mGTer->RDiscToR2(Pt2dr(0,0)),
        mGTer->RDiscToR2(Pt2dr(1,1))- mGTer->RDiscToR2(Pt2dr(0,0)),
        Pt2dr(0,0),
        Pt2dr(mRSrIm,mRSrIm)

    ),
    mPDV   (aPDV),
    mGeoI  (aPDV.Geom()),
    mSzTer (mSimIntBxTer.sz()),
    mImT   (mSzTer.x,mSzTer.y),
    mTImT  (mImT),
    mDataZ (mImT.data()),
    mBoxIm (   Pt2di(-1,-1)+round_down(Pt2dr(mSimCurBxIn._p0)/mRSrIm),
               Pt2di( 1, 1)+round_up(Pt2dr(mSimCurBxIn._p1)/mRSrIm)
           ),
    mInterpTabul (new cTplCIKTabul<float,double>(12,12,mPrIm->BicubParam().Val())),
    mInterp      (mInterpTabul),
    mInterpIm    (new cInterpolBicubique<U_INT1>(mPrIm->BicubParam().Val())),
    mSzK         (mInterp->SzKernel()+1),
    mBoxTerInterp  (mSimIntBxTer._p0+Pt2di(mSzK,mSzK),mSimIntBxTer._p1-Pt2di(mSzK,mSzK))
{
   ELISE_COPY
   (
        mImT.all_pts(),
           trans(mTifMnt->in(),mSimIntBxTer._p0)*mSimFOM.ResolutionAlti() 
        +  mSimFOM.OrigineAlti() ,
        mImT.out()
   );
}

 // mSimFOM.OrigineAlti()+ mSimZMin*mSimFOM.ResolutionAlti();
/*
         Box2di                       mBoxTerInterp;
         int                          mSzK;
         mSimIntBxTer
*/

    //======================================================================
    //======================================================================
    //======================================================================


void cAppliMICMAC::SimulateOneBoxPDV(cPriseDeVue & aPDV,Tiff_Im * aFileIm,Tiff_Im * aFileMNT)
{

    mSimCurBxTer = aPDV.Geom().BoxImageOfBox
                   (
                       Pt2dr(mSimCurBxIn._p0) ,Pt2dr(mSimCurBxIn._p1),
                       1.0,false,&mSimZMin,&mSimZMax,
                       3.0* mSimRPlani
                   );
     Pt2di aP0 = mGTer->R2ToDisc(mSimCurBxTer._p0);
     Pt2di aP1 = mGTer->R2ToDisc(mSimCurBxTer._p1);

     Box2di aBox(aP0,aP1);
     Box2di aBoxMnt(Pt2di(0,0),mSimSzMNT);

     if (InterVide(aBox,aBoxMnt))
     {
        return;
     }


     mSimIntBxTer = Inf(aBox,aBoxMnt);

   cMMSimulZBuf aZB(*this,aPDV);

   Pt2di aOffset_Out_00;
   float aDef = -1e10;
   double anEpsII = 1e-7 *  mSimRPlani;
   aZB.SetEpsilonInterpoleInverse(anEpsII);
   Im2D_REAL4 aRes =  mPrIm->ReprojInverse().Val()                                                           ?
                      aZB.BasculerAndInterpoleInverse(aOffset_Out_00,mSimIntBxTer._p0,mSimIntBxTer._p1,aDef) :
                      aZB.Basculer(aOffset_Out_00,mSimIntBxTer._p0,mSimIntBxTer._p1,aDef)                    ;






   std::cout 
             // << " IM-in "  << mSimCurBxIn._p0 << mSimCurBxIn._p1 
             // << " IM-out "  << mSimCurBxOut._p0 << mSimCurBxOut._p1 
             // << " TER "  << mSimCurBxTer._p0 << mSimCurBxTer._p1 
             // << " ITER "  << mSimIntBxTer._p0 << mSimIntBxTer._p1 
             // << " RES "  <<  aRes.sz() << aOffset_Out_00
             << " RES "  <<    aOffset_Out_00 << aRes.sz() 
             << "\n";

   if (aFileIm)
   {
       TIm2D<float,double> aTRes(aRes);
       Pt2di aSzTer = mSimIntBxTer.sz();
       Im2D_U_INT1 aImTer(aSzTer.x,aSzTer.y) ;
       ELISE_COPY
       (
            aImTer.all_pts(),
            trans(mTifText->in(),mSimIntBxTer._p0),
            aImTer.out()
       );
       U_INT1 ** aDataImTer = aImTer.data();

       Pt2di aSzIm = aRes.sz();
       Im2D_U_INT1 aImProj(aSzIm.x,aSzIm.y,0);
       TIm2D<U_INT1,INT> aTImProj(aImProj);
       Pt2di aPIm;

/*
for (double aD=-10 ; aD <=300 ; aD+=5.0)
{
   std::cout << " D " << aD <<   " " << int(El_CTypeTraits<U_INT1>::TronqueR(aD)) << "\n";
}
*/
       for (aPIm.x=0 ; aPIm.x<aSzIm.x ; aPIm.x++)
       {
           for (aPIm.y=0 ; aPIm.y<aSzIm.y ; aPIm.y++)
           {
                // Pt3dr aPTer = aZB.InverseProjDisc(Pt3dr(aPIm.x,aPIm.y,aTRes.get(aPIm)));
                Pt3dr aPTer = aZB.InverseProjDisc(aPIm);
                Pt2dr aPt2(aPTer.x,aPTer.y);
                if (aZB.mBoxTerInterp.inside(Pt2di(aPt2)))
                {
                   aPt2  = aPt2-Pt2dr(mSimIntBxTer._p0);
                   double aVr = aZB.mInterpIm->GetVal(aDataImTer,aPt2);
                   //U_INT1 aVi = El_CTypeTraits<U_INT1>::TronqueR(aVr);
                   // aVi = aVr;
                   int aVi = ElMax(1,ElMin(255,round_ni(aVr)));
                   aTImProj.oset(aPIm,aVi);
                }
           }
       }
       double aSzK =mPrIm->SzFTM().Val();
       Fonc_Num aF = StdFoncChScale
                    (
                        (aImProj.in(0)),
                        -Pt2dr(aOffset_Out_00)       ,
                        Pt2dr(1/mRSrIm,1/mRSrIm),
                        Pt2dr(aSzK,aSzK)
                    ) ;
       if (mPrIm->Bruit().IsInit())
       {
           aF = Max(0,Min(255,aF+mPrIm->Bruit().Val()*frandr()));
       }
       ELISE_COPY(rectangle(mSimCurBxOut),aF,aFileIm->out());
   }


   if (aFileMNT)
   {
      
       Fonc_Num aF = StdFoncChScale
                    (
                        (aRes.in(0)),
                        -Pt2dr(aOffset_Out_00)       ,
                        Pt2dr(1/mRSrIm,1/mRSrIm)
                    ) ;
       ELISE_COPY
       (
            rectangle(mSimCurBxOut),
            (aF-mSimFOM.OrigineAlti())/mSimFOM.ResolutionAlti(),
           aFileMNT->out()
       );
   }



}

    //======================================================================

void cAppliMICMAC::SimulatePDV(cPriseDeVue & aPDV)
{

   if (! mPrIm->PatternSel()->Match(aPDV.Name()))
      return;
   std::cout << "DOING  " << aPDV.Name() << " " << mRSrIm << "\n";

   std::cout << "SZ " << aPDV.SzIm() << "\n";
   int aSzBloc = mPrIm->SzBloc().Val();
   int aBrd = mPrIm->SzBrd().Val();
   Pt2di aBPrd(aBrd,aBrd);

   Tiff_Im * aFileMNT=0;
   if (mPrIm->KeyProjMNT().IsInit())
   {
      std::string aNameMNT =  WorkDir()+ mICNM->Assoc1To1(mPrIm->KeyProjMNT().Val(),aPDV.Name(),true);
      std::cout << aNameMNT << "\n";
      aFileMNT = new Tiff_Im
                         (
                              aNameMNT.c_str(),
                              aPDV.SzIm(),
                              GenIm::real4,
                              Tiff_Im::No_Compr,
                              Tiff_Im::BlackIsZero
                         );
   }

   Tiff_Im * aFileIm=0;
   if (mPrIm->KeyIm().IsInit())
   {
      std::string aNameIm =  WorkDir()+ mICNM->Assoc1To1(mPrIm->KeyIm().Val(),aPDV.Name(),true);
      std::cout << aNameIm << "\n";
      aFileIm  = new Tiff_Im
                         (
                              aNameIm.c_str(),
                              aPDV.SzIm(),
                              GenIm::u_int1,
                              Tiff_Im::No_Compr,
                              Tiff_Im::BlackIsZero
                         );
   }

   cDecoupageInterv2D  aDI2d
                       (
                           Box2di(Pt2di(0,0),aPDV.SzIm()),
                           Pt2di(aSzBloc,aSzBloc),
                           Box2di(-aBPrd,aBPrd)
                        );

   for (int aKBox=0 ; aKBox<aDI2d.NbInterv() ; aKBox++)
   {
       mSimCurBxIn = aDI2d.KthIntervIn(aKBox);
       mSimCurBxOut = aDI2d.KthIntervOut(aKBox);
       SimulateOneBoxPDV(aPDV,aFileIm,aFileMNT);
   }

   std::cout << " -- DONE  " << aPDV.Name() << "\n";
   delete aFileMNT;
}

    //======================================================================
    //======================================================================
    //======================================================================

void cAppliMICMAC::GenerateSimulations() 
{
    mRRGlob = 1.0;
    if (Planimetrie().IsInit())
       mRRGlob = Planimetrie().Val().RatioResolImage().Val();

    mSSim = &(SectionSimulation().Val());
    mSRel = &(mSSim->SimulRelief());
    mPrIm = &(mSSim->ProjImPart());

    mRSrIm = mPrIm->RatioSurResol().ValWithDef(mRRGlob);

    std::cout << "cAppliMICMAC::GenerateSimulations \n";

    ELISE_ASSERT
    (
       mEtapesMecComp.size()==1,
       "Plusieurs etapes en GenerateSimulations()"
    );
    OneEtapeSetCur(*mEtapesMecComp.back());

    // std::cout <<  mEtapesMecComp.size() << " " << mCurEtape << " " << LoadTer() << "\n";
    // std::cout << "NBPx " <<   LoadTer()->NbPx() << "\n";


    ELISE_ASSERT
    (
        mDimPx==1,
       "Plusieurs paralaxe en GenerateSimulations()"
    );


    mSimFOM = OrientFromOneEtape(*mCurEtape);
    mSimFactPA  = ElAbs(mSimFOM.ResolutionPlani().x/mSimFOM.ResolutionAlti());
    mSimRPlani = dist8(mSimFOM.ResolutionPlani());
    mGTer = &(mCurEtape->GeomTer());

   bool isNew;
   mTifMnt = new Tiff_Im(mCurEtape->KPx(0).FileIm(isNew));
   ELISE_ASSERT(mTifMnt->type_el()==GenIm::real4,"Bad MNT type in Simul");
   mSimSzMNT = mTifMnt->sz();

   SimulateRelief(isNew);

   std::string aNameFullText = WorkDir() + mSSim->ImRes();

   if (! ELISE_fp::exist_file(aNameFullText))
   {
       std::cout << "DOING  TEXTURE\n";
       Im2DGen aIm = Tiff_Im::UnivConvStd(WorkDir() + mSSim->Texton()).ReadIm();

       Pt2di aSz = aIm.sz();

       Fonc_Num aFx = Min(FX%(2*aSz.x),2*aSz.x-1-FX%(2*aSz.x));
       Fonc_Num aFy = Min(FY%(2*aSz.x),2*aSz.y-1-FY%(2*aSz.y));

       Tiff_Im::Create8BFromFonc
       (
             aNameFullText,
             mTifMnt->sz(),
             aIm.in()[Virgule(aFx,aFy)]
       );
       std::cout << "---- DONE TEXTURE\n";
   }
   mTifText = new Tiff_Im(Tiff_Im::BasicConvStd(aNameFullText));

   std::cout << "DOING  INTERVAL Z \n";
   ELISE_COPY(mTifMnt->all_pts(),mTifMnt->in(),VMax(mSimZMax)|VMin(mSimZMin));
   mSimZMin = mSimFOM.OrigineAlti()+ mSimZMin*mSimFOM.ResolutionAlti();
   mSimZMax = mSimFOM.OrigineAlti()+ mSimZMax*mSimFOM.ResolutionAlti();
   std::cout << " ---   DONE Interval Z [" << mSimZMin << "," << mSimZMax << "]\n";

   for 
   (
       tContPDV::iterator itP =   mPrisesDeVue.begin();
       itP !=   mPrisesDeVue.end();
       itP++
   )
   {
       SimulatePDV(**itP);
       // std::cout << "DOING  " << (*itP)->Name() << "\n";
       // std::cout << " -- DONE  " << (*itP)->Name() << "\n";
   }


   delete mTifMnt;
   delete mTifText;
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
