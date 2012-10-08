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
#include "general/all.h"
#include "MICMAC.h"

/*
 @  Pour l'instant  ZMin-ZMax sont globaux au chantier;
==================================================== 
void cAppliMICMAC::MakePartiesCachees()
   Fonction globale calcule ZMin-ZMax pour le chantier et 
  parcours tous les  prise de vue pour appeler MakePartiesCachees(cPriseDeVue & ...)

==================================================== 
void cAppliMICMAC::MakePartiesCachees ( cPriseDeVue & aPdv, 
const cGenerePartiesCachees & aGPC, double aZMin, double aZMax)
*/





namespace NS_ParamMICMAC
{

// static bool DEBUGPC = true;

ElAffin2D AffPix2Ter(const cFileOriMnt & anOri)
{
   Pt2dr aResol = anOri.ResolutionPlani();
   return ElAffin2D
          (
               anOri.OriginePlani(),
               Pt2dr(aResol.x,0),
               Pt2dr(0,aResol.y)
          );
}


ElAffin2D   AffPixTer2BoxLoc(const Box2di & aBox)
{
      return ElAffin2D(-Pt2dr(aBox._p0),Pt2dr(1,0),Pt2dr(0,1));
}


/*********************************************/
/*                                           */
/*              cMicMacZbuf                  */
/*                                           */
/*********************************************/

class cMicMacZbuf : public cZBuffer
{
    public :
         cMicMacZbuf
         (
               const  cAppliMICMAC &,
               bool                  aDoDequant,
               const cEtapeMecComp * aCurEtape,
               const Pt2di &         aTrGT,   // Translation entre la BOX GT et ImZ,ImMasq
               const cGeomDiscFPx&,
               cPriseDeVue & aPdv,
               Im2D_INT2  aImZInit,
               Im2D_Bits<1> aImMasq
         );
         void Inspect();
         double ZBrutOfXY(const Pt2di & aP)   const { return   mTImTer.get(aP); }
         Im2D_REAL4 ZBrutOfXY()   const { return   mImTer; }
        bool SelectP(const Pt2di & aP)   const  ;
    private  :
        Pt3dr ProjTerrain(const Pt3dr &) const;
        double ZofXY(const Pt2di & aP)   const ; 
        double ZInterpofXY(const Pt2dr & aP,bool & OK) const;



        bool SelectPBascul(const Pt2dr & aP)   const;


        const  cAppliMICMAC &            mAppli;
        Pt2di                mTrGT;
        const cGeomDiscFPx & mGeoT;
        cGeomImage &     mGeom;
        Im2D_REAL4        mImTer;
        TIm2D<REAL4,REAL8>  mTImTer;
        Im2D_Bits<1>     mImMasq;
        TIm2DBits<1>     mTImMasq;
};


Pt3dr cMicMacZbuf::ProjTerrain(const Pt3dr & aPTer) const
{
   // On n'utilise pas la methode BasculeInv, qui devrait a peu pres
   // le faire car elle est redefinie

    double aPax[theDimPxMax] = {0,0};
    aPax[0] = aPTer.z;
    Pt2dr aPIm = mGeom.Objet2ImageInit_Euclid(Pt2dr(aPTer.x,aPTer.y),aPax);
/*
    if (mAppli.InversePx())
       return Pt3dr(aPIm.x,aPIm.y,1.0/aPTer.z);
   else 
*/
    Pt3dr aRes = Pt3dr(aPIm.x,aPIm.y,aPTer.z/mGeoT.RatioResAltiPlani());
   
    return aRes;
}
 


cMicMacZbuf::cMicMacZbuf 
(
      const  cAppliMICMAC & anAppli,
      bool                  aDoDequant,
      const cEtapeMecComp * aCurEtape,
      const Pt2di &         aTrGT,
      const cGeomDiscFPx& aGeomTer,
      cPriseDeVue & aPdv,
      Im2D_INT2  aImZInit,
      Im2D_Bits<1> aImMasq
) :
  cZBuffer
  (
        aGeomTer.RDiscToR2(Pt2dr(0,0)),
        aGeomTer.RDiscToR2(Pt2dr(1,1))- aGeomTer.RDiscToR2(Pt2dr(0,0)),
        Pt2dr(0,0),
        Pt2dr(aCurEtape->DeZoomTer(),aCurEtape->DeZoomTer())
  ) ,
  mAppli  (anAppli),
  mTrGT   (aTrGT),
  mGeoT   (aGeomTer),
  mGeom   (aPdv.Geom()),
  mImTer  (aGeomTer.SzClip().x,aGeomTer.SzClip().y),
  mTImTer (mImTer),
  mImMasq (aImMasq),
  mTImMasq (mImMasq)
{



  ELISE_COPY(mImTer.all_pts(),trans(aImZInit.in(),mTrGT),mImTer.out());

  if (aDoDequant)
  {
     ElImplemDequantifier aDeq(mImTer.sz());
     aDeq.SetTraitSpecialCuv(false);
     aDeq.DoDequantif(mImTer.sz(), mImTer.in(),true);
     ELISE_COPY
     (
          mImTer.all_pts(),
          aDeq.ImDeqReelle(),
          mImTer.out()
     );
  }
}

double cMicMacZbuf::ZInterpofXY(const Pt2dr & aP,bool & OK) const
{
   double aDef = -1e10;
   double aZ = mTImTer.getr(aP,aDef);
   OK = (aZ!=aDef);
   if (! OK)
     return aDef;
   mGeoT.PxDisc2PxReel(&aZ,&aZ);
   return aZ;
}

double cMicMacZbuf::ZofXY(const Pt2di & aP)   const 
{
   double aZ = mTImTer.get(aP);
   mGeoT.PxDisc2PxReel(&aZ,&aZ);
   return aZ;
}


bool cMicMacZbuf::SelectP(const Pt2di & aP)   const  
{
   return mTImMasq.get(aP+mTrGT,0);
   // return mTImMasq.get(aP,0);
}

bool cMicMacZbuf::SelectPBascul(const Pt2dr & aP)   const
{
   return   true;
}


void cMicMacZbuf::Inspect()
{
    static Video_Win aW3 = Video_Win::WStd(Pt2di(1000,900),1);

    ELISE_COPY(aW3.all_pts(),0,aW3.ogray());

    for (int anX = 0 ; anX<mGeoT.SzClip().x ; anX++)
        for (int anY = 0 ; anY<mGeoT.SzClip().y ; anY++)
        {
              Pt2di aP(anX,anY);
              aW3.draw_circle_loc(Pt2dr(aP),1.0,aW3.pdisc()(SelectP(aP) ? P8COL::red : P8COL::green));
        }
    
/*
    ELISE_COPY
    (
       mImTer.all_pts(),
       mImTer.in(),
       aW3.ocirc()
    );
    ELISE_COPY
    (
       select(mImMasq.all_pts(),!mImMasq.in()),
       128,
       aW3.ogray()
    );

    while (1)
    {
       Clik   aCl = aW3.clik_in();
       TIm2D<REAL4,REAL8> aTit(mImTer);

       std::cout << "    " << aCl._pt  << mImTer.sz() << mImMasq.sz() << "\n";
       std::cout << mTImTer.get(aCl._pt)  << " " 
                 << mTImMasq.get(round_ni(aCl._pt))  << " " 
                 << "\n";
    }
*/
}


/*********************************************/
/*                                           */
/*              cAppliMICMAC                 */
/*                                           */
/*********************************************/

bool OrthoEgalePC(const cGenerePartiesCachees & aGPC)
{
   if (! aGPC.MakeOrthoParImage().IsInit())
     return true;
   const cMakeOrthoParImage &  aMOI = aGPC.MakeOrthoParImage().Val();
   if 
   (
              // aMOI.ResolRelOrhto().IsInit() 
              (aMOI.ResolRelOrhto().ValWithDef(1.0) !=1.0)
           || aMOI.ResolAbsOrtho().IsInit() 
           || aMOI.PixelTerrainPhase().IsInit()
   )
       return false;
   return true;
}

std::string  cAppliMICMAC::NamePC
             (
                 bool ForPC,
                 const cGenerePartiesCachees & aGPC,
                 cEtapeMecComp * anEtape,
                 const cPriseDeVue & aPdv
             ) const
{
   if (aGPC.KeyCalcPC().IsInit()  &&( (OrthoEgalePC(aGPC) || (! ForPC))))
   {
      std::string aRes;
      if (aGPC.AddChantierKPC().Val())
      {
          aRes = WorkDir() + ICNM()->Assoc1To2(aGPC.KeyCalcPC().Val(),aPdv.Name(),NameChantier(),true);
      }
      else
      {
          aRes = WorkDir() + ICNM()->Assoc1To1(aGPC.KeyCalcPC().Val(),aPdv.Name(),true);
      }
      if (aGPC.SupresExtChantierKPC().Val())
         aRes= StdPrefix(aRes);
      return aRes;
   }
   return    FullDirResult() 
          +  std::string("MasqPC_")
          +  (  ((!OrthoEgalePC(aGPC))&&(! ForPC)) ? "Ortho_":"")
          +  NameChantier()
          +  std::string("_Num" ) + ToString(anEtape->Num()) +std::string("_")
          +  StdPrefix(aPdv.Name()) ;
}

Box2di  BoxTer2Disc
        (
             const cGeomDiscFPx & aGT,
             const Box2dr & aBoxTer,
             const Pt2di  & PRab
        )
{

   Pt2di aProvP0 = aGT.R2ToDisc(aBoxTer._p0); 
   Pt2di aProvP1 = aGT.R2ToDisc(aBoxTer._p1); 

   Pt2di aRab(3,3);
   return Box2di
          (
              Sup(Pt2di(0,0),Inf(aProvP0,aProvP1)-aRab),
              Inf(aGT.SzDz(),Sup(aProvP0,aProvP1)+aRab)
          );
}

void cAppliMICMAC::MakePartiesCachees
                   (
                       cPriseDeVue & aPdv,
                       const cGenerePartiesCachees & aGPC,
                       double aZMin,
                       double aZMax
                   )
{


   if (aGPC.FiltreName().IsInit() && (! aGPC.FiltreName().Val()->Match(aPdv.Name())))
   {
         return;
   }
   bool FullIm = aGPC.ImSuperpMNT().Val();
   // std::cout << "MakePC  : " << aPdv.Name() << "\n";

/*
    Trois cas de figure :
      - Pas d'ortho :
           MEC/PC   
      - Une ortho superposable 
           MEC/Ortho 

      - Une ortho non superp

          -  MEC/PC     :  force non ortho
          -  MEC/Ortho  :  force ortho

   Trois mode Force Ortho   , Force Non Ortho , Adapt
*/

   bool DoOrtho =  aGPC.MakeOrthoParImage().IsInit();


   std::string   anEntete = NamePC(true,aGPC,mCurEtape,aPdv);
   std::cout << "ENTETE =" << anEntete << "\n";
   if (aGPC.DoOnlyWhenNew().Val())
   {
       if (ELISE_fp::exist_file(anEntete+".tif"))
       {
          std::cout << "Fichier existe, ne fait rien\n";
          return;
       }
   }

   Box2dr aBoxTer =  aPdv.Geom().EmpriseTerrain(&aZMin,&aZMax,0.0);



   cGeomDiscFPx   aGT = mCurEtape->GeomTer();
   aGT.SetClipInit();
   Box2dr aBoxGlob(aGT.P0(), aGT.P1());

   cMetaDataPartiesCachees aMetaData;
   aMetaData.Done() = false;

    if (InterVide(aBoxGlob,aBoxTer))
    {
       // std::cout << "INTER VIDE : " << aPdv.Name() << "\n";
       MakeFileXML ( aMetaData, anEntete+".xml");
       return;
    }



   double aZMoy;

   // On reduit l'intervalle de Z en l'adaptant sur la Box deja calculee (en theorie on peu iterer)
   if (! aGPC.ZMoy().IsInit())
   {
      Box2di aB  = BoxTer2Disc(aGT,aBoxTer,Pt2di(0,0));
      GetIntervZ(aB,aZMin,aZMax,aZMoy);
      aBoxTer =  aPdv.Geom().EmpriseTerrain(&aZMin,&aZMax,0.0);
   }
  

    if (InterVide(aBoxGlob,aBoxTer))
    {
       MakeFileXML ( aMetaData, anEntete+".xml");
       return;
    }

   std::cout << "PC-Name " << aPdv.Name() << " Z " << aZMin << " - " << aZMax<<  "\n";


   const cFilePx & aFP = mCurEtape->KPx(0);
   Tiff_Im     aTFP =   aFP.FileIm() ;
   GenIm::type_el aTypePx = aTFP.type_el();


   Box2di aB  = BoxTer2Disc(aGT,aBoxTer,Pt2di(3,3));
   Pt2di aP0 = aB._p0;
   Pt2di aP1 = aB._p1;

   Im2D_INT2 aImZ(1,1);
   Im2D_Bits<1>  aImMasq(1,1);

   if (FullIm)
   {
          aP0 = Pt2di(0,0);
          aP1 = aTFP.sz();
   }


    // On va calculer le masque des points terrain qui sont dans le masque terrain
    // initial et dont la projection au Z calcule est dans l'image 
    // Eventuellement on adpate la boite pour la reduire
   

   if  ((aP0.x<aP1.x) && (aP0.y <aP1.y))
   {
      cGeomImage & aGeoI =  aPdv.Geom();
      aGT.SetClip(aP0,aP1);
      Pt2di aSzIm = aP1- aP0;

      aImZ = Im2D_INT2(aSzIm.x,aSzIm.y);
      TIm2D<INT2,INT> aTImZ(aImZ);
      aImMasq = Im2D_Bits<1>(aSzIm.x,aSzIm.y,0);
      Tiff_Im aTFM = FileMasqOfResol(mCurEtape->DeZoomTer());

      TIm2DBits<1>     aTImMasq(aImMasq);
      ELISE_COPY
      (
         rectangle(Pt2di(0,0),aSzIm),
         trans(Virgule(aTFP.in(0),aTFM.in(0)),aP0),
         Virgule(aImZ.out(),aImMasq.out())
      );

      Pt2di aNewP0 = aSzIm;
      Pt2di aNewP1 (0,0);
      //int aPaxI[theDimPxMax] = {0,0};
      double aPaxR[theDimPxMax] = {0,0};
      Pt2di aSzPdv = aPdv.SzIm();

      cLineariseProj aLP;
      int aSzBloc = 7;

      // Calcul le masque en geometrie image
      for (int aY0=0; aY0<aSzIm.y ; aY0+=aSzBloc)
      {
          int aY1 = ElMin(aSzIm.y,aY0+aSzBloc);
          for (int aX0=0; aX0<aSzIm.x ; aX0+=aSzBloc)
          {
             int aX1 = ElMin(aSzIm.x,aX0+aSzBloc);

             int aSZ=0;
             int aNbZ=0;
             {
                Pt2di aP;
                for (aP.y =aY0 ; aP.y<aY1 ; aP.y++)
                {
                   for (aP.x=aX0 ; aP.x<aX1 ; aP.x++)
                   {
                       if (aTImMasq.get(aP))
                       {
                           aNbZ++;
                           aSZ += aTImZ.get(aP);
                       }
                   }
                }
             }

             if (aNbZ!=0)
             {
                Pt2di aP;
                aSZ /= aNbZ;
                aPaxR[0] = aSZ;
                aGT.PxDisc2PxReel(aPaxR,aPaxR);
                Pt2di aPMil((aX0+aX1)/2,(aY0+aY1)/2);

                Pt2dr aPImC =  aGeoI.Objet2ImageInit_Euclid(aGT.DiscToR2(aPMil),aPaxR);
                Pt2dr aDxIm =  aGeoI.Objet2ImageInit_Euclid(aGT.DiscToR2(aPMil+Pt2di(1,0)),aPaxR) -aPImC;
                Pt2dr aDyIm =  aGeoI.Objet2ImageInit_Euclid(aGT.DiscToR2(aPMil+Pt2di(0,1)),aPaxR) -aPImC;

                aPaxR[0] = aSZ+1;
                aGT.PxDisc2PxReel(aPaxR,aPaxR);
                Pt2dr aDzIm =  aGeoI.Objet2ImageInit_Euclid(aGT.DiscToR2(aPMil),aPaxR)-aPImC;


                for (aP.y =aY0 ; aP.y<aY1 ; aP.y++)
                {
                   for (aP.x=aX0 ; aP.x<aX1 ; aP.x++)
                   {
                      Pt2dr aPImCur =  aGT.DiscToR2(aP);
                      if (aTImMasq.get(aP) && aGeoI.IsInMasqAnam(aPImCur))
                      {
                         int aZ = aTImZ.get(aP);
                         Pt2dr aPIm2 =   aPImC
                                          + aDxIm * (aP.x-aPMil.x)
                                          + aDyIm * (aP.y-aPMil.y)
                                          + aDzIm * (aZ-aSZ);

                         if (
                                     (aPIm2.x>0)
                                &&   (aPIm2.y>0)
                                &&   (aPIm2.x<aSzPdv.x)
                                &&   (aPIm2.y<aSzPdv.y)
                              )
                         {
                            aNewP0 = Inf(aNewP0,aP);
                            aNewP1 = Sup(aNewP1,aP);
                         }
                         else
                         {
                            aTImMasq.oset(aP,0);
                         }
                      }
                   }
                }
             }
          }
      }



       
      if (0)
      {
         double aDZ = 0.5;
         static Video_Win aW = Video_Win::WStd(Pt2di(1200,1200),aDZ);
         aW.clear();
         ELISE_COPY(aImMasq.all_pts(),aImMasq.in(),aW.odisc());
         aW.draw_rect(Box2dr(aNewP0,aNewP1),aW.pdisc()(P8COL::red));
      }

      Pt2di aNewSz = aNewP1-aNewP0;
      if ((!FullIm) && (aNewSz.x>0) &&  (aNewSz.y>0))
      {
          Im2D_INT2 aNewImZ(aNewSz.x,aNewSz.y);
          Im2D_Bits<1>  aNewImMasq(aNewSz.x,aNewSz.y);

          ELISE_COPY
          (
              aNewImZ.all_pts(),
              trans(Virgule(aImZ.in(),aImMasq.in()),aNewP0),
              Virgule(aNewImZ.out(),aNewImMasq.out())
          );
          aImZ= aNewImZ;
          aImMasq = aNewImMasq;
      }
      aP1 = aP0+ aNewP1;
      aP0 = aP0+aNewP0;
   }

   if (FullIm)
   {
          aP0 = Pt2di(0,0);
          aP1 = aTFP.sz();
   }



   if ((aP0.x>=aP1.x) || (aP0.y>=aP1.y))
   {
       MakeFileXML ( aMetaData, anEntete+".xml");

       return;
   }

   Pt2di aP0Glob = aP0;
   Pt2di aP1Glob = aP1;
   
   int aSzBord = aGPC.SzBord().Val();
   int aMaxSz= aGPC.SzBloc().Val();


   Pt2di aPBord(aSzBord,aSzBord);
   cDecoupageInterv2D  aDI2d(Box2di(aP0Glob,aP1Glob),Pt2di(aMaxSz,aMaxSz),Box2di(-aPBord,aPBord));



   std::string aStrEnt = anEntete+".tif";
   Tiff_Im  aTifPC
            (
               aStrEnt.c_str(),
               aP1-aP0,
               GenIm::u_int1,
               Tiff_Im::No_Compr,
               Tiff_Im::BlackIsZero
            );

   aMetaData.Done() = true;
   aMetaData.Offset()= aP0Glob;
   aMetaData.Sz()= aTifPC.sz();
   double aPas = aGPC.PasDisc().Val();
   aMetaData.Pas()= aPas;
   aMetaData.SeuilUse() = round_ni(aGPC.SeuilUsePC() / aPas);



// std::cout << "KBOX " <<  aP0 << aP1 << "\n";

   Tiff_Im *  aTifPCO = 0;
   for (int aKBox=0 ; aKBox<aDI2d.NbInterv() ; aKBox++)
   {
        Box2di  aBoxIn = aDI2d.KthIntervIn(aKBox);

        aGT.SetClip(aBoxIn._p0,aBoxIn._p1);


        cMicMacZbuf aMmZB
               (
                       *this,
                       aGPC.Dequant().ValWithDef(type_im_integral(aTypePx)),
                       mCurEtape,
                       aBoxIn._p0-aP0Glob,
                       aGT,
                       aPdv,
                       aImZ,
                       aImMasq
                );
       // aMmZB.Inspect();

        aMmZB.SetWithBufXYZ(aGPC.BufXYZ().Val());

        float aZDef = -1e15;

        Im2D_REAL4 aIPC = aMmZB.ZCaches (Pt2di(0,0),aGT.SzClip(), aZDef);
    

        double aZoom = aGT.ResolZ1();


       Box2di  aBoxOut = aDI2d.KthIntervOut(aKBox);
       if (DoOrtho)
       {
           cMakeOrthoParImage aMOPI = aGPC.MakeOrthoParImage().Val();

           cFileOriMnt anOriMNT = OrientFromOneEtape(*mCurEtape);
           cFileOriMnt anOriOrtho = anOriMNT;

           double aRx = anOriOrtho.ResolutionPlani().x;
           double aRy = anOriOrtho.ResolutionPlani().y;
           double aRXSurY = aRx/aRy;

           ELISE_ASSERT(ElAbs(ElAbs(aRXSurY)-1)<1e-5,"Incoherence in ResolAbsOrtho");
           double aResRel =  aMOPI.ResolRelOrhto().ValWithDef(1.0);
           if (aMOPI.ResolAbsOrtho().IsInit())
           {
               aResRel = aMOPI.ResolAbsOrtho().Val()/ElAbs(aRx);
           }

           anOriOrtho.ResolutionPlani() = Pt2dr(aRx,aRy)*aResRel;


           if (aMOPI.PixelTerrainPhase().IsInit())
           {
               Pt2dr  aPixP = aMOPI.PixelTerrainPhase().Val();
               double aResol =  anOriOrtho.ResolutionPlani().x; 

               Pt2di aPK = round_ni((anOriOrtho.OriginePlani()-aPixP) / aResol);
               anOriOrtho.OriginePlani() = aPixP + Pt2dr(aPK) * aResol;
           }

           anOriOrtho.NameFileMasque().SetNoInit();
           anOriOrtho.NameFileMnt() = "NO";

           if (aGT.SzClip()!=aBoxIn.sz())
           {
               std::cout  << "WARN ORTHO : " << aBoxIn.sz()  << aGT.SzClip() << "\n";
           }

           // Similitude envoyant un pixel MNT vers un pixel Ortho

           ElAffin2D aAfPM2PO  = AffPix2Ter(anOriOrtho).inv() * AffPix2Ter(anOriMNT);


           Box2di aBoxMntGlob(aP0Glob,aP1Glob);
           Box2di aBoxOrthoGlob = R2I(aBoxMntGlob.BoxImage(aAfPM2PO));

           Box2di  aBoxOrthoIn  =  Inf(aBoxOrthoGlob,R2I(aBoxIn.BoxImage(aAfPM2PO)));
           Box2di  aBoxOrthoOut =  Inf(aBoxOrthoGlob,R2I(aBoxOut.BoxImage(aAfPM2PO)));

           anOriOrtho.NombrePixels() = round_ni(Pt2dr(anOriMNT.NombrePixels())/aResRel);

           ElAffin2D  aAfPML2POL =   AffPixTer2BoxLoc(aBoxOrthoIn) * aAfPM2PO   * AffPixTer2BoxLoc(aBoxIn).inv();

           Im2D_Bits<1> aMasqOrt = aMmZB.ImOkTer();
           Im2D_U_INT1 aIPCOrt(1,1);
           if (! OrthoEgalePC(aGPC))
           {
                Pt2di aSzO = aBoxOrthoIn.sz();
                aMasqOrt = Im2D_Bits<1>(aSzO.x,aSzO.y);
                TIm2DBits<1> aTMasqO(aMasqOrt);
                TIm2DBits<1>  aTMasqM(aMmZB.ImOkTer());

                aIPCOrt = Im2D_U_INT1(aSzO.x,aSzO.y);
                TIm2D<U_INT1,INT> aTIPCOrt(aIPCOrt);
                TIm2D<REAL4,REAL8> aTIPC(aIPC);

                ElAffin2D aAfPOL2PML  = aAfPML2POL.inv();
                Pt2di aPO;

                double aMul = aZoom *  aGPC.PasDisc().Val();
                for (aPO.x=0 ; aPO.x<aSzO.x; aPO.x++)
                {
                    for (aPO.y=0 ; aPO.y<aSzO.y; aPO.y++)
                    {
                        Pt2dr aPM = aAfPOL2PML(Pt2dr(aPO));
                        aTMasqO.oset(aPO,aTMasqM.get(round_ni(aPM),0)==1);
                        double aVal = aTIPC.getr(aPM,0) / aMul;
                        aTIPCOrt.oset(aPO,ElMax(0,ElMin(255,int(aVal))));
                    }
               }
          }

           
           /// Geometrie Masq Ter / Image PC

           MakeOrtho
           (
                   aBoxOrthoGlob,
                   aBoxOrthoIn,
                   aBoxOrthoOut,
                   //  aBoxMntGlob,aBoxIn,
                   aKBox,
                   aZMoy,
                   aMasqOrt,
                   aAfPML2POL,
                   aMOPI, // aGPC.MakeOrthoParImage().Val(),
                   aPdv,
                   aMmZB,
                   aMetaData
           );
           if (! OrthoEgalePC(aGPC))
           {
               Pt2di OP0G = aBoxOrthoGlob._p0;
               if (aKBox==0)
               {
                   std::string   anEnteteO = NamePC(false,aGPC,mCurEtape,aPdv);
std::cout << "XXXXX " << anEnteteO << "\n";

                   cMetaDataPartiesCachees aMTDO = aMetaData;
                   aMTDO.Offset() = OP0G;
                   aMTDO.Sz() = aBoxOrthoGlob.sz();
                   MakeFileXML (aMTDO, anEnteteO+".xml");

                   std::string aStrEntO = anEnteteO+".tif";
                   aTifPCO = new Tiff_Im
                        (
                            aStrEntO.c_str(),
                            aBoxOrthoGlob.sz(),
                            GenIm::u_int1,
                            Tiff_Im::No_Compr,
                            Tiff_Im::BlackIsZero
                         );

               }
               Fonc_Num aFPC =   Min(254,(aIPCOrt.in())) *aMasqOrt.in() + 255 * (1-aMasqOrt.in());
               ELISE_COPY
               (
                    rectangle(aBoxOrthoOut._p0- OP0G,aBoxOrthoOut._p1-OP0G),
                    trans(aFPC, OP0G-aBoxOrthoIn._p0),
                    aTifPCO->out()
               );
           }

           if (aKBox==0)
           {
                std::string aDir = mICNM->Dir()+aMOPI.DirOrtho().Val();
                ELISE_fp::MkDir(aDir);
                MakeFileXML(anOriOrtho, aDir+aMOPI.FileMTD().Val());

                if (aMOPI.MakeMTDMaskOrtho().IsInit())
                {
                    const cMakeMTDMaskOrtho & aMMMO = aMOPI.MakeMTDMaskOrtho().Val();
                    MakeFileXML(aMMMO.Mesures(),aDir+aMMMO.NameFileSauv().Val());
                }
           }
       }


       if ( OrthoEgalePC(aGPC))
       {
           Fonc_Num aFPC =   Min(254,(aIPC.in()/aZoom)/aGPC.PasDisc().Val()) * aMmZB.ImOkTer().in()
                           + 255 * (1-aMmZB.ImOkTer().in());
           ELISE_COPY
           (
                rectangle(aBoxOut._p0- aP0Glob,aBoxOut._p1-aP0Glob),
                trans(aFPC, aP0Glob-aBoxIn._p0),
                aTifPC.out()
           );
       }

   }

   MakeFileXML ( aMetaData, anEntete+".xml");


   if (DoOrtho)
   {
      DeleteAndClear(mOrthoImIn);
      DeleteAndClear(mIntOrth);
   }


}



/*



*/


static inline Pt2dr ToIm(const Pt3dr & aPProj , const Pt2dr & aTrIm, const double & aScIm, const Pt2dr & aP0Out)
{
    Pt2dr aProj2(aPProj.x,aPProj.y);
    aProj2  =  aProj2 * aScIm - aTrIm;

    return  aProj2 +  aP0Out;
}




void cAppliMICMAC::MakeOrtho
     (
          const Box2di &             aBoxGlob,
          const Box2di&              aBoxIn,
          const Box2di&              aBoxOut,
          int                        aKBox,
          double                     aZMoyen,
          Im2D_Bits<1>               aMasqT,
          const  ElAffin2D  &        aAfPML2POL ,
          const cMakeOrthoParImage & aMOPI,
          cPriseDeVue & aPDV,
          cMicMacZbuf & aZB,
          cMetaDataPartiesCachees & aMDPC
     )
{
  

    ElAffin2D  aAfPOL2PML = aAfPML2POL.inv();

    Pt2di aSzTGlob = aBoxGlob._p1-aBoxGlob._p0;

    Pt2di aSzT = aBoxIn.sz();


    std::string aNameIn = WorkDir() + ICNM()->Assoc1To1(aMOPI.KeyCalcInput(),aPDV.Name(),true);
    if (!ELISE_fp::exist_file(aNameIn))
    {
       for (int aK=0 ; aK<3 ; aK++)
          std::cout << "----NO FILE " << aNameIn << " for ortho " << "\n";
       std::cout << "\n";
       return;
    }
    Tiff_Im aFIn = Tiff_Im::StdConvGen(aNameIn.c_str(),aMOPI.NbChan().Val(),false);
    int aDzTer = mCurEtape->DeZoomTer();

    int aNbC = aFIn.nb_chan();
    std::vector<Im2DGen *> mOrthos;
    Pt2di aSzImIn = aFIn.sz();
    cGeomImage & aGeom = aPDV.Geom();

    Box2dr aBoxIm(Pt2dr(0,0),Pt2dr(aSzImIn));
    double aIntMax =  aBoxIm.Interiorite(aBoxIm.milieu());

    Im2D_REAL4  aImIncH (1,1);
    TIm2D<REAL4,REAL8> aTImIncH(aImIncH);
    bool doImIncH = false;
    bool doIncZMoy = false;
    bool doIncByFront = false;
    std::string aNameIncH;
    if (aMOPI.KeyCalcIncidHor().IsInit())
    {
        aImIncH.Resize(aSzT);
        aTImIncH =  TIm2D<REAL4,REAL8>(aImIncH);
        aNameIncH =  WorkDir() + ICNM()->Assoc1To1(aMOPI.KeyCalcIncidHor().Val(),aPDV.Name(),true);
        doIncByFront = aMOPI.ImageIncIsDistFront().Val();
        doImIncH = true;
        doIncZMoy= aMOPI.CalcIncAZMoy().ValWithDef(!doIncByFront);
    }


    Output anOutLoadImIn = Output::onul();
    Fonc_Num aFoncOrth=0;

    for (int aKC=0 ; aKC<aNbC ; aKC++)
    {
        if (aKBox==0)
        {
           mOrthoImIn.push_back(Ptr_D2alloc_im2d(aFIn.type_el(),aSzImIn.x,aSzImIn.y));
           mIntOrth.push_back(mOrthoImIn.back()->BiCubIm(aMOPI.OrthoBiCub().Val(),aMOPI.ScaleBiCub().Val()));
           anOutLoadImIn = (aKC==0) ?  mOrthoImIn.back()->out() : Virgule(anOutLoadImIn, mOrthoImIn.back()->out());
        }

        mOrthos.push_back(Ptr_D2alloc_im2d(aFIn.type_el(),aSzT.x,aSzT.y));
        aFoncOrth = (aKC==0) ? mOrthos.back()->in() : Virgule(aFoncOrth,mOrthos.back()->in());
    }


    if (aKBox==0)
    {
       Fonc_Num aFoncIn = aFIn.in_proj();

       ELISE_COPY
       (
            rectangle(Pt2di(0,0),aSzImIn),
            aFoncIn,
            anOutLoadImIn
       );
    }



    double aScIm =  aMOPI.ResolIm().Val() * aDzTer;
    Pt2dr aTrIm  =  Pt2dr(aMOPI.TranslateIm().Val());

    Pt2dr  aP0Out = Pt2dr(aZB.P0_Out()) * aScIm;


    Pt2di aPO;
    TIm2DBits<1> aTMasqT(aMasqT);
    int aRF = aMOPI.RepulsFront().Val();

    for (aPO.x=0 ; aPO.x<aSzT.x ; aPO.x++)
    {
        double aValOut = -1.1e10;
        double aValTest = -1e10;
        for (aPO.y=0 ; aPO.y<aSzT.y ; aPO.y++)
        {
            if (aTMasqT.get(aPO))
            {
                Pt2dr aPM  = aAfPOL2PML(Pt2dr(aPO));
                bool aOK;
                Pt3dr aPTer = aZB.ProjReelle(aPM,aOK);

                if (aOK)
                {
                    Pt2dr aPIm = ToIm(aPTer,aTrIm,aScIm,aP0Out);

                    for (int aKC=0 ; aKC<aNbC ; aKC++)
                    {
                        double aVal = mIntOrth[aKC]->GetDef(aPIm,aValOut);
                        if (aVal < aValTest)
                        {
                            aVal = 0;
                            aTMasqT.oset(aPO,0);
                        }
                        // mOrthos[aKC]->SetR(aP,aVal);
                        mOrthos[aKC]->TronqueAndSet(aPO,aVal);
                    }
                    if (doImIncH)
                    {
                       Pt2dr aPImInc = aPIm;
                       if (doIncZMoy)
                       {
                          aPImInc = ToIm(aZB.ProjDisc(Pt3dr(aPM.x,aPM.y,aZMoyen)),aTrIm,aScIm,aP0Out);
                       }
                       if (doIncByFront)
                       {
                           double aD  = (aIntMax-aBoxIm.Interiorite(aPImInc)) * (1.0/aIntMax) ;
                           aTImIncH.oset(aPO,aD);
                       }
                       else
                       {
                          ElSeg3D  aSeg = aGeom.FaisceauPersp(aPImInc * aDzTer);
                          double aTeta = acos(-aSeg.TgNormee().z);
                          if (aRF > 0)
                          {
                              aTeta += ElMax(0.0,(aRF-aBoxIm.Interiorite(aPImInc))*(10.0/aRF));
                          }
                          aTImIncH.oset(aPO,aTeta);
                       }
                     }
                 } 
            } 
            else
            {
                for (int aKC=0 ; aKC<aNbC ; aKC++)
                    mOrthos[aKC]->TronqueAndSet(aPO,0);
                if (doImIncH)
                {
                   aImIncH.SetR(aPO,3.14);
                }
            }
        }
    }


    std::string aNameOrtho = WorkDir() + ICNM()->Assoc1To1(aMOPI.KeyCalcOutput(),aPDV.Name(),true);


     Tiff_Im  aTifOrtho = (aKBox==0)                     ?
                          Tiff_Im 
                          (
                              aNameOrtho.c_str(),
                              aSzTGlob,
                              aFIn.type_el(),
                              Tiff_Im::No_Compr,
                              aFIn.phot_interp()
                           )                             :
                           Tiff_Im ( aNameOrtho.c_str()) ;


       ELISE_COPY
       (
            rectangle(aBoxOut._p0- aBoxGlob._p0,aBoxOut._p1-aBoxGlob._p0),
            trans(aFoncOrth, aBoxGlob._p0-aBoxIn._p0),
            aTifOrtho.out()
       );



    if (doImIncH)
    {
        double aSsR = aMOPI.SsResolIncH().Val();
        aMDPC.SsResolIncH().SetVal(aSsR);
        Tiff_Im aFileIncH  = (aKBox==0)                    ?
                             Tiff_Im
                             (
                                aNameIncH.c_str(),
                                round_up(Pt2dr(aSzTGlob)/aSsR),
                                GenIm::real4,
                                Tiff_Im::No_Compr,
                                Tiff_Im::BlackIsZero
                             )                             :
                             Tiff_Im(aNameIncH.c_str())    ;

         Pt2di aP0r = round_down(Pt2dr(aBoxOut._p0- aBoxGlob._p0)/aSsR);
         Pt2di aP1r = round_up(Pt2dr(aBoxOut._p1- aBoxGlob._p0)/aSsR);
         Pt2dr aTrr = Pt2dr(aBoxGlob._p0-aBoxIn._p0) ;

         // std::cout << "Verif Trr ?? " << aTrr << "\n";

         ELISE_COPY
         (
               rectangle(aP0r,aP1r),
               StdFoncChScale(aImIncH.in_proj(),aTrr,Pt2dr(aSsR,aSsR)),
               aFileIncH.out()
         );

/*
         ELISE_COPY
         (
               aFileIncH.all_pts(),
               StdFoncChScale(aImIncH.in_proj(),Pt2dr(0,0),Pt2dr(aSsR,aSsR)),
               aFileIncH.out()
         );
*/
    }
    DeleteAndClear(mOrthos);

}




void cAppliMICMAC::GetIntervZ(const Box2di & aBox,double & aZMin,double & aZMax,double & aZMoy)
{
   ELISE_ASSERT
   (
       mDimPx==1,
       "Gestion parties cachees : requiert Dim Px = 1"
   );

  const cFilePx & aFP = mCurEtape->KPx(0);
  Tiff_Im     aTFP =   aFP.FileIm() ;

  Tiff_Im aTFM = FileMasqOfResol(mCurEtape->DeZoomTer());

  Symb_FNum  aFZ  (Rconv(aTFP.in()));
  Symb_FNum  aFM  (Rconv(aTFM.in(0)));


  Symb_FNum  aFMZ (aFM * aFZ);
  Symb_FNum  aCFM (1-aFM);
  // double aZMin,aZMax;

  // Pour que les hors masques (CFM=Complementaire Fonction Masque)
  // n'aient pas d'influence on leurs donne des valeurs hors borne

  Pt2di aP0 = Sup(Pt2di(0,0),aBox._p0);
  Pt2di aP1 = Inf(aTFP.sz(),aBox._p1);

  double  aSomMZ,aSomM;
  ELISE_COPY
  (
       rectangle(aP0,aP1),
       Virgule
       (
              aFMZ + aCFM *  1e9,
              aFMZ + aCFM * -1e9,
              aFMZ,
              aFM
       ),
       Virgule ( VMin(aZMin),VMax(aZMax),sigma(aSomMZ),sigma(aSomM))
  );


  aZMoy = aSomMZ / aSomM;

  const cGeomDiscFPx &  aGT = mCurEtape->GeomTer();
  aGT.PxDisc2PxReel(&aZMin,&aZMin);
  aGT.PxDisc2PxReel(&aZMax,&aZMax);
  aGT.PxDisc2PxReel(&aZMoy,&aZMoy);
}

void cAppliMICMAC::MakePartiesCachees()
{
   if (! Paral_Pc_NbProcess().IsInit())
   {
      if (CalledByProcess().Val()) 
         return;
      if (DoNothingBut().IsInit()  && (! ButDoPartiesCachees().Val()) && (! ButDoOrtho().Val()))
      {
         if (! DoMEC().Val())
            return;
      }
   }
   cTplValGesInit<cGenerePartiesCachees> aVI_PC =  mCurEtape->EtapeMEC().GenerePartiesCachees();
   if (! aVI_PC.IsInit())
      return;

   const cGenerePartiesCachees & aGPC = aVI_PC.Val();

    if (! aGPC.UseIt().Val())
      return;

   if (DoNothingBut().IsInit())
   {
       if ( ButDoOrtho().Val() && (!ButDoPartiesCachees().Val()))
       {
            if (!aGPC.MakeOrthoParImage().IsInit())
               return;
       }
   }

  const cFilePx & aFP = mCurEtape->KPx(0);
  Tiff_Im     aTFP =   aFP.FileIm() ;
  double aZMin,aZMax,aZMoy;

  if (aGPC.ZMoy().IsInit())
  {
     aZMin = aZMax = aZMoy = aGPC.ZMoy().Val();
  }
  else
  {
    GetIntervZ(Box2di(Pt2di(0,0),aTFP.sz()),aZMin,aZMax,aZMoy);
  }

/*
*/

   cSetName * aSetN = ICNM()->KeyOrPatSelector(aGPC.PatternApply().Val());


   if (aVI_PC.Val().ByMkF().Val() && (! Paral_Pc_IdProcess().IsInit()))
   {
       // Si on demande une mise en // il faut etre logique !
       int aNbP = ElMax(2,ByProcess().Val());
       std::list<std::string> aLP;
       for (int aKP=0 ; aKP < aNbP ; aKP++)
       {
            std::string aCom  =  PrefixGenerikRecalEtapeMicmMac(*mCurEtape)
                               + std::string(" DoMEC=0")
                               + std::string(" Paral_Pc_NbProcess=") + ToString(aNbP)
                               + std::string(" Paral_Pc_IdProcess=") + ToString(aKP)   ;
            aLP.push_back(aCom);
       }
       ExeProcessParallelisable(true,aLP);
   }
   else
   {
      int aCpt=0;
      for 
      (
          tContPDV::const_iterator itPdv=mPrisesDeVue.begin();
          itPdv!=mPrisesDeVue.end();
          itPdv++
      )
     {

          if (aSetN->IsSetIn((*itPdv)->Name()))
          {
             bool DoIt = true;
             if (Paral_Pc_IdProcess().IsInit())
             {
                DoIt = ((aCpt%Paral_Pc_NbProcess().Val()) == Paral_Pc_IdProcess().Val());
             }
// std::cout << "IDPIsni" << Paral_Pc_IdProcess().IsInit() i
// << " DoIt:: " << DoIt << " " << (*itPdv)->Name()  << "\n";
             if (DoIt) 
                MakePartiesCachees(**itPdv,aGPC,aZMin,aZMax);

             aCpt++;
          }
     }
     // getchar();
   }
  

}


///////////////////////////////////////////////////

};

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
