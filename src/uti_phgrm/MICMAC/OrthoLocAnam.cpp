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



Pt3dr cAppliMICMAC::ToRedr
      (
             const cFileOriMnt & aFOMInit,
             const cFileOriMnt &aFOMCible,
             const Pt3dr &aPDiscInit
      )
{
   Pt3dr aPTerInit =  ToMnt(aFOMInit,aPDiscInit);
   Pt3dr aPTerRedr =  mAnamSA->ToOrLoc(aPTerInit);
 
   return FromMnt(aFOMCible,aPTerRedr);
}

void cAppliMICMAC::MakeRedrLocAnamSA()
{
//FOM : lire XML;

   const cEtapeMEC &  anET = mCurEtape->EtapeMEC();
   if (! anET.RedrLocAnam().IsInit())
      return;
   const cRedrLocAnam & aRLA =  anET.RedrLocAnam().Val();

   if (! mAnamSA) return;
   if (! mAnamSA->HasOrthoLoc()) return;

   if (! mAnamSA->OrthoLocIsXCste())
   {
      cElWarning::OrhoLocOnlyXCste.AddWarn
      (
            "For file anam : " + mNameAnamSA,
           __LINE__,
           __FILE__

      );
      return;
   }

   bool Show =true;
    // ==================================

   std::string aNamePxTif = mCurEtape->KPx(0).NameFile();
   Tiff_Im aFilePx = Tiff_Im::StdConv(mCurEtape->KPx(0).NameFile());


  bool AddAutoM = aRLA.UseAutoMask().Val();

   //Tiff_Im aFM = 

   double aPxExtr[2];
   Symb_FNum aFPx(aFilePx.in());
   Symb_FNum aFM(mCurEtape->FoncMasqIn(AddAutoM) * 1e7);

   if (Show) std::cout << "BEGIN LEC GLOB\n";
   ELISE_COPY
   (
       aFilePx.all_pts(),
       Virgule(aFPx-aFM,aFPx+aFM),
       Virgule(VMax(aPxExtr[1]),VMin(aPxExtr[0]))
   );
   if (Show) std::cout << "END LEC GLOB\n";

   cFileOriMnt aFOMInit = StdGetObjFromFile<cFileOriMnt>
                          (
                               StdPrefix(aNamePxTif)+".xml",
                               StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                              "FileOriMnt",
                              "FileOriMnt"
                          );
   //mCurEtape->RemplitOri(aFOMInit);

   // std::cout << "NAME FIPX " << aFilePx.name() << "\n";
   // std::cout <<  "Int Px [ " << aPxExtr[0] << " " << aPxExtr[1] << "]\n";

   // Pt2di aPtSupDisc(-1e9,-1e9);
   // Pt2di aPtInfDisc(+1e9,+1e9);

   cFileOriMnt aFOMFinale = aFOMInit;

   {
       Pt2dr aPtSupR(-1e20,-1e20);
       Pt2dr aPtInfR(+1e20,+1e20);
       Box2dr aBoxInit(Pt2dr(0,0),Pt2dr(aFOMInit.NombrePixels()));

       Pt2dr aCor[4] ;
       aBoxInit.Corners(aCor);

       for (int aKP = 0 ; aKP<4 ; aKP++)
       {
           for (int aKZ=0 ; aKZ<2 ; aKZ++)
           {
               Pt3dr aPDiscInit(aCor[aKP].x,aCor[aKP].y,aPxExtr[aKZ]);
               Pt3dr aPTerInit =  ToMnt(aFOMInit,aPDiscInit);
               Pt3dr aPTerRedr =  mAnamSA->ToOrLoc(aPTerInit);
               //  Pt3dr aPDiscRedr = FromMnt(aFOMCible,aPTerRedr);

               aPtSupR.SetSup(Proj(aPTerRedr));
               aPtInfR.SetInf(Proj(aPTerRedr));
           }
       }

       // std::cout << "  BOX INIT "  << aBoxInit._p0  << " " << aBoxInit._p1 << "\n";
       // std::cout << "  BOX REDR "  << aPtInfR  << " " << aPtSupR << "\n";


       Pt2dr aRPlan = aFOMFinale.ResolutionPlani();
       double aRy = aRPlan.y;
       double aRx = aRPlan.x;
       Pt2dr aOri ( 
                     (aRx>0) ?   aPtInfR.x : aPtSupR.x,
                     (aRy>0) ?   aPtInfR.y : aPtSupR.y
                  );

       
       Pt2dr aP0  =  FromMnt(aFOMFinale,aPtInfR);
       Pt2dr aP1  =  FromMnt(aFOMFinale,aPtSupR);

       Pt2dr aPInfD = Inf(aP0,aP1);
       Pt2dr aPSupD = Sup(aP0,aP1);

       Pt2di aNb = round_up(aPSupD)-round_down(aPInfD);
       aFOMFinale.NombrePixels() = aNb;
       aFOMFinale.OriginePlani() = aOri;
       // std::cout << "WWWWWWWWWWWW " << aNb << aOri << "\n";

       // std::cout << " VVVV " << FromMnt(aFOMFinale,aPtInfR) << FromMnt(aFOMFinale,aPtSupR)  << " NB " << aNb << "\n";
    }


    // cFileOriMnt aFOMFinale = aFOMInit;
    // aFOMFinale.NombrePixels() = aPtSup-aPtInf;


    Pt2di aSzInit = aFOMInit.NombrePixels();
    Pt2di aSzFinale = aFOMFinale.NombrePixels();


    std::string aNameOut = FullDirResult()+aRLA.NameOut() + ".tif";
    std::string aNameMaskOut = FullDirResult()+aRLA.NameMasq() + ".tif";
    MakeFileXML(aFOMFinale, StdPrefix(aNameOut)+".xml");


    if (aRLA.NameNuage().IsInit())
    {
       std::string aNNRed = aRLA.NameNuage().Val();
       std::string aNNInit =    mCurEtape->NameXMLNuage();
       cXML_ParamNuage3DMaille aNuageInit = StdGetFromSI(aNNInit,XML_ParamNuage3DMaille);
       cXML_ParamNuage3DMaille aNuageFinal = aNuageInit;
       cXmlOrthoCyl * anOC = 0;
       bool DoExp = false;

       // On regarde si il y a des cas que l'on sait gerer 
       {
          cXmlOneSurfaceAnalytique * aSAN = aNuageInit.Anam().PtrVal();
          if (aSAN)
          {
             anOC = aSAN->XmlDescriptionAnalytique().OrthoCyl().PtrVal();
          }
       }


       // En fait le seul cas que l'on sache gerer est le cylindre
       if (anOC)
       {
          DoExp = true;
          aNuageFinal.Anam().SetNoInit();
          aNuageFinal.RepereGlob().SetVal(anOC->Repere());
       }
       else if (mAnaGeomMNT)
       {
            cXmlModeleSurfaceComplexe aModele= StdGetFromSI(mNameAnamSA,XmlModeleSurfaceComplexe);
            aNuageFinal.Anam() = SFromId(aModele,"TheSurfAux");
            DoExp = true;
       }

       if (DoExp)
       {
          aNuageFinal.NbPixel() = aFOMFinale.NombrePixels() ;
          aNuageFinal.Image_Profondeur().Val().Image() = aRLA.NameOut() + ".tif";
          aNuageFinal.Image_Profondeur().Val().Masq() = aRLA.NameMasq() + ".tif";
          ElAffin2D anAffC2M = ElAffin2D::TransfoImCropAndSousEch(aFOMFinale.OriginePlani(),aFOMFinale.ResolutionPlani());
          aNuageFinal.Orientation().OrIntImaM2C().SetVal( El2Xml(anAffC2M)); // RPCNuage

          MakeFileXML(aNuageFinal,FullDirResult() +aNNRed);
       }
    }

// std::cout << "AAAAAAAAAbbGGgg " << aRLA.NameOriGlob() << "\n"; getchar();

    int  aZoom = mCurEtape->DeZoomTer();
    cFileOriMnt aFomR1 =  aFOMFinale;
    aFomR1.NombrePixels() = aFomR1.NombrePixels()  * aZoom  ;
    aFomR1.ResolutionPlani() =  aFomR1.ResolutionPlani()  / double(aZoom);
    aFomR1.ResolutionAlti() =  aFomR1.ResolutionAlti()  / double(aZoom);
    MakeFileXML(aFomR1, FullDirResult() + aRLA.NameOriGlob());

    if (Show) std::cout << "BEFIN CREAT FIL\n";
    Tiff_Im aFileRes
            (
                 aNameOut.c_str(),
                 aSzFinale,
                 GenIm::real4,
                  Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero
             );
    Tiff_Im aFileMaskOut
            (
                 aNameMaskOut.c_str(),
                 aSzFinale,
                 GenIm::bits1_msbf,
                 //GenIm::u_int1,
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero
             );

    //Tiff_Im * aTLAB = new Tiff_Im

    if (Show) std::cout << "END  CREAT FIL\n";
             
    int aRabX = aRLA.XRecouvrt().Val();
    double aMem = aRLA.MemAvalaible().Val();

    int aMulLarg = (int)( aRLA.FilterMulLargY().Val() );
    int aNbIter = (int)( aRLA.NbIterFilterY().Val() );
    int aMoyFin = aRLA.FilterXY().Val();
    int aNbIterMF = aRLA.NbIterXY().Val();

    double aSeuilHaut = aRLA.DensityHighThresh().Val();
    double aSeuilBas  = aRLA.DensityLowThresh().Val();

    double aSzX = aMem / aSzFinale.y;
    int aNbX = round_up(aSzFinale.x/aSzX);

    // double aMaxEtirY= 0.0;
   // double aRx = ElAbs(aFOMInit.ResolutionPlani().x);
   // double aRZ = ElAbs(aFOMInit.ResolutionAlti());
   // double aZRatio =   aRZ / aRx;

    //double aRegulPente = 1.0;


    for (int aKX=0 ; aKX < aNbX; aKX++)
    {

        // if (Show) std::cout <<  "KKKKKK " << aKX << " " << aNbX << "\n";
    // Cacul des tailles 
         int aX0_Out = round_ni((aSzFinale.x * double(aKX)) / aNbX);
         int aX1_Out = round_ni((aSzFinale.x * double(aKX+1)) / aNbX);
         int aX0_In = ElMax(0,aX0_Out-aRabX);
         int aX1_In = ElMin(aSzFinale.x,aX1_Out+aRabX);
         int aSzXIn = aX1_In-aX0_In;

         Pt2di aSzIn(aSzXIn,aSzInit.y);
         Pt2di aSzInRedr(aSzXIn,aSzFinale.y);

    // Lecture des input

         Im2D_REAL4 aImZIn(aSzIn.x,aSzIn.y,0.0);
         TIm2D<REAL4,REAL8> aTZIn(aImZIn);
         Im2D_Bits<1>  aImMasqIn(aSzIn.x,aSzIn.y,0);
         TIm2DBits<1> aTMasqIn(aImMasqIn);

         ELISE_COPY
         (
               aImZIn.all_pts(),
               trans(aFilePx.in_proj(),Pt2di(aX0_In,0)),
               aImZIn.out()
         );
         ELISE_COPY
         (
               aImMasqIn.all_pts(),
               trans(mCurEtape->FoncMasqIn(AddAutoM),Pt2di(aX0_In,0)),
               aImMasqIn.out()
         );


    // Generation de l'image de compteurs
         Im2D_REAL4 aImCpt(aSzInRedr.x,aSzInRedr.y,0.0);
         TIm2D<REAL4,REAL8> aTCpt(aImCpt);
         Im2D_REAL4 aImZOut(aSzInRedr.x,aSzInRedr.y,0.0);
         TIm2D<REAL4,REAL8> aTZOut(aImZOut);
         Im1D_REAL4 aImBuf(aSzInRedr.y);
         float * aDBuf = aImBuf.data();

         Im2D_U_INT1 aImLabel(aSzInRedr.x,aSzInRedr.y,0);
         TIm2D<U_INT1,INT> aTLab(aImLabel);

         for (int anX = 0 ; anX < aSzIn.x; anX++)
         {
              Im1D_REAL4 aImLarg(aSzInRedr.y,1.0);
              float * aDLarg = aImLarg.data();

              std::vector<double> aVY;
              std::vector<double> aVZ;
              std::vector<double> aVLarg;
              for (int anY=0 ; anY<aSzIn.y ; anY++)
              {
                   double aZ = aTZIn.get(Pt2di(anX,anY));
                   Pt3dr aPDisc(aX0_In+anX,anY,aZ);
                   Pt3dr aPRedr = ToRedr(aFOMInit,aFOMFinale,aPDisc);
                   Pt3dr aQRedr = ToRedr(aFOMInit,aFOMFinale,aPDisc+Pt3dr(0,1,0));

                   double aDY = ElAbs(aQRedr.y -aPRedr.y);
                   aVY.push_back(aPRedr.y);
                   aVZ.push_back(aPRedr.z);
                   aVLarg.push_back(aDY);

                   if (aTMasqIn.get(Pt2di(anX,anY)))
                      aTCpt.incr(Pt2dr(anX,aPRedr.y),aDY);
              }

              aImLarg.raz();
              for (int anY= 0 ; anY <(aSzIn.y-1) ; anY++)
              {
                  double aYr0 = aVY[anY];
                  double aYr1 = aVY[anY+1];
                  int aYI0 = ElMax(0,round_up(aYr0));
                  int aYI1 = ElMin(aSzInRedr.y,round_up(aYr1));

                  double aF = FromSzW2FactExp(aVLarg[anY]*aMulLarg,aNbIter);
                  for (int y=aYI0 ; y<aYI1; y++)
                  {
                      aDLarg[y] = (float)aF;
                      double aP1 = (y-aYr0) / (aYr1-aYr0);
                      aTZOut.oset
                      (
                          Pt2di(anX,y),
                          aVZ[anY] * (1-aP1) +  aVZ[anY+1]  * aP1
                      );
                  }
              }


              for (int anY= 0 ; anY <aSzInRedr.y ; anY++)
              {
                  aDBuf[anY] = (float)( aTCpt.get(Pt2di(anX,anY)) );
              }
              FilterMoyenneExpVar(aDBuf,aDLarg,aSzInRedr.y,aNbIter);
              for (int anY= 0 ; anY <aSzInRedr.y ; anY++)
              {
                  aTCpt.oset(Pt2di(anX,anY),aDBuf[anY]);
              }
         } 


    // Fitrage de type hysteresis

         Fonc_Num aF = aImCpt.in_proj();
         for (int aK=0 ; aK < aNbIterMF; aK++)
             aF = rect_som(aF,aMoyFin) / ElSquare(1+2.0*aMoyFin);
         ELISE_COPY(aImCpt.all_pts(),aF,aImCpt.out());

         for (int anX = 0 ; anX < aSzIn.x; anX++)
         {
              for (int anY= 0 ; anY <aSzInRedr.y ; anY++)
              {
                  double aVal = aTCpt.get(Pt2di(anX,anY));
                  if (aVal<aSeuilBas) 
                     aTLab.oset(Pt2di(anX,anY),0);
                  else if (aVal<aSeuilHaut) 
                    aTLab.oset(Pt2di(anX,anY),1);
                  else 
                    aTLab.oset(Pt2di(anX,anY),2);
              }
         }

         ELISE_COPY(aImLabel.border(1),0,aImLabel.out());
         Neigh_Rel RV4(Neighbourhood::v4());
         ELISE_COPY
         (
             conc
             (
                select
                (
                    select(aImLabel.all_pts(),(aImLabel.in()==1)),
                    RV4.red_max(aImLabel.in()==0)
                ),
                aImLabel.neigh_test_and_set(Neighbourhood::v4(),1,0,10)
             ),
             0,
             aImLabel.out()
         );

    // Sauvegarde des resultats

         ELISE_COPY
         (
               rectangle(Pt2di(aX0_Out,0),Pt2di(aX1_Out,aSzInRedr.y)),
               trans(aImZOut.in(),Pt2di(-aX0_In,0)),
               aFileRes.out()
         );

         ELISE_COPY
         (
               rectangle(Pt2di(aX0_Out,0),Pt2di(aX1_Out,aSzInRedr.y)),
               trans(aImLabel.in()!=0,Pt2di(-aX0_In,0)),
               aFileMaskOut.out()
         );
    }

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
