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

namespace NS_ParamMICMAC
{

class cBasculeMNT : public cZBuffer
{
     public :
         cBasculeMNT
	 (
	       cGeomDiscFPx & aGeomT,
               Pt2dr OrigineIn,
               Pt2dr StepIn,
               Pt2dr OrigineOut,
               Pt2dr StepOut,
	       const cGeomBasculement3D * aGeom,
	       bool  UseF,
	       float ** aDataF,
	       INT2 **  aDataI,
	       Pt2di    aSzData
	 ) :
             cZBuffer(OrigineIn,StepIn,OrigineOut,StepOut),
	     mGeomT (aGeomT),
             mGeom (aGeom),
             mUseF (UseF),
             mDataF (aDataF),
             mDataI (aDataI),
             mBicF  (-0.5),
             mBicI  (-0.5),
	     mSincF (3),
	     mSincI (3),
	     mInterpF (&mBicF),
	     mInterpI (&mBicI),
	     mSzData  (aSzData)
         {
         }
     private :
         Pt3dr ProjTerrain(const Pt3dr & aP) const
	 {
             return mGeom->Bascule(aP);
	 }

         Pt3dr InvProjTerrain(const Pt3dr & aP) const
	 {
             return mGeom->BasculeInv(aP);
	 }


         double ZofXY(const Pt2di & aP)  const
         {
              static double aPaxIn[theDimPxMax] ={0,0};
              static double aPaxOut[theDimPxMax];
	      
               aPaxIn[0]= mUseF ? mDataF[aP.y][aP.x] : mDataI[aP.y][aP.x];
	       mGeomT.PxDisc2PxReel(aPaxOut,aPaxIn);
	       return aPaxOut[0];

         }

	 double ZInterpofXY(const Pt2dr & aP,bool & OK) const
	 {
              static double aPaxIn[theDimPxMax] ={0,0};
              static double aPaxOut[theDimPxMax];
	      
	       int aNb = mInterpF->SzKernel();
	       OK =     (aP.x > aNb) 
	             && (aP.y > aNb)
	             && (aP.x < mSzData.x-aNb-1) 
	             && (aP.y < mSzData.y-aNb-1) ;
               if (! OK)
	          return 0;
               aPaxIn[0]= mUseF ? mInterpF->GetVal(mDataF,aP) : mInterpI->GetVal(mDataI,aP);
	       mGeomT.PxDisc2PxReel(aPaxOut,aPaxIn);
	       return aPaxOut[0];
	 }

	 cGeomDiscFPx &       mGeomT;
         const cGeomBasculement3D * mGeom;
	 bool                 mUseF;
	 float **             mDataF;
	 INT2 **              mDataI;

         cInterpolBicubique<float>      mBicF;
         cInterpolBicubique<INT2>       mBicI;
         cInterpolSinusCardinal<float>  mSincF;
         cInterpolSinusCardinal<INT2>   mSincI;

	 cInterpolateurIm2D<float> *    mInterpF;
	 cInterpolateurIm2D<INT2> *     mInterpI;
	 Pt2di                          mSzData;
};

void cEtapeMecComp::OneBasculeMnt
     (
           Pt2di aP0Sauv,
           Pt2di aP1Sauv,
           cBasculeRes & aBR, 
           float ** aDataF,
           INT2 ** aDataI,
	   Pt2di   aSzData
    )
{
    ELISE_ASSERT
    (
         mIsOptimCont,
        "Basculement requiert une optimisation continue "
    );
 
    cFileOriMnt anOri;
    if (aBR.Explicite().IsInit())
    {
       anOri = aBR.Explicite().Val();
       std::string aNameXML =    mAppli.FullDirResult()
                          + StdPrefixGen(anOri.NameFileMnt())
			  + std::string(".xml");
        MakeFileXML(anOri,aNameXML);
    }
    else if (aBR.ByFileNomChantier().IsInit())
    {
       std::string aNameFile = 
                   mAppli.WorkDir()
               +   aBR.Prefixe() 
               +   (aBR.NomChantier().Val() ? mAppli.NameChantier() :"")
	       +   aBR.Postfixe();


       anOri = StdGetObjFromFile<cFileOriMnt>
               (
	            aNameFile,
	            mAppli.NameSpecXML(),
		    aBR.NameTag().Val(),
		    "FileOriMnt"

	       );
    }
    else
    {
       ELISE_ASSERT(false,"Internal Error cEtapeMecComp::OneBasculeMnt");
    }
   // cFileOriMnt * aPtrOri=0;
   // cFileOriMnt * aPtrOri=0;
   
   // cFileOriMnt & anOri = aBR.Ori();

    // std::cout << "XML MADE \n"; getchar();

   const cGeomBasculement3D * aGeomB = 0;

   if (anOri.Geometrie() == eGeomMNTEuclid)
   {
       if (
                 (mAppli.GeomMNT()==eGeomMNTFaisceauIm1ZTerrain_Px1D)
              || (mAppli.GeomMNT()==eGeomMNTFaisceauIm1ZTerrain_Px2D)
	   )
       {
           aGeomB = (mAppli.PDV1()->Geom().GeoTerrainIntrinseque());
       }
/*
 CE CAS PARTICULIER VIENT DE CE QUE cGeomImage_Faisceau redifinit la methode
Bascule. A ete utilise avec Denis Feurer & Co pour basculer en terrain
les reultat image.
*/
       else if (
                 (mAppli.GeomMNT()==eGeomMNTFaisceauIm1PrCh_Px1D)
              || (mAppli.GeomMNT()==eGeomMNTFaisceauIm1PrCh_Px2D)
	   )
       {
           aGeomB = &(mAppli.PDV2()->Geom());
       }
       else
       {
           ELISE_ASSERT(false,"Geometrie source non traitee dans le basculement");
       }
   }
   else
   {
       ELISE_ASSERT(false,"Geometrie destination non traitee dans le basculement");
   }

   Pt2dr  aP0 = mGeomTer.DiscToR2(Pt2di(0,0));
   Pt2dr  aP1 = mGeomTer.DiscToR2(Pt2di(1,1));


   cBasculeMNT   aBasc
                 (
                     mGeomTer,
                     aP0,
                     aP1-aP0,
                     anOri.OriginePlani(),
                     anOri.ResolutionPlani(),
                     aGeomB,
                     mIsOptimCont,
                     aDataF,
                     aDataI,
		     aSzData
		 );

 Pt2di anOffset;
 double aDef = -1e10;
 double aSousDef = -9e9;

 std::cout << "BEGIN BASCULE \n";
 //Im2D_REAL4   aMnt=  aBasc.Basculer(anOffset,aP0Sauv,aP1Sauv,aDef);
 Im2D_REAL4   aMnt=  aBasc.BasculerAndInterpoleInverse(anOffset,aP0Sauv,aP1Sauv,aDef);


 ELISE_COPY
 (
     select(aMnt.all_pts(),aMnt.in() > aSousDef),
     (aMnt.in()-anOri.OrigineAlti())/anOri.ResolutionAlti(),
     aMnt.out()
 );

 std::cout  << anOffset << " " << aMnt.sz();
 std::cout << "END  BASCULE \n";

 bool isNewFile;
 Tiff_Im aFileRes = Tiff_Im::CreateIfNeeded
         (
             isNewFile,
	     anOri.NameFileMnt(),
	     anOri.NombrePixels(),
	     GenIm::real4,
	     Tiff_Im::No_Compr,
	     Tiff_Im::BlackIsZero
	 );
 Tiff_Im * aFileMasq =0;
 if ( anOri.NameFileMasque().IsInit())
 {
         aFileMasq = new Tiff_Im(Tiff_Im::CreateIfNeeded
                         (
                             isNewFile,
	                     anOri.NameFileMasque().Val(),
	                     anOri.NombrePixels(),
	                     GenIm::bits1_msbf,
	                     Tiff_Im::No_Compr,
	                     Tiff_Im::BlackIsZero
	                     )
			 );
   }

   if (isNewFile)
   {
      ELISE_COPY
      (
         aFileRes.all_pts(),
         aBR.OutValue().Val(),
         aFileRes.out()
      );
      if (aFileMasq)
      {
         ELISE_COPY
         (
            aFileMasq->all_pts(),
            0,
            aFileMasq->out()
         );
      }
   }

   Im2D_REAL4 anOld(aMnt.sz().x,aMnt.sz().y);

   ELISE_COPY
   (
       anOld.all_pts(),
       trans(aFileRes.in(aBR.OutValue().Val()),anOffset),
       anOld.out()
   );
   ELISE_COPY
   (
       select(anOld.all_pts(),aMnt.in()>aSousDef),
       aMnt.in(),
       anOld.out()
   );
   ELISE_COPY
   (
       rectangle(anOffset,anOffset+aMnt.sz()),
       trans(anOld.in(),-anOffset),
       aFileRes.out()
   );

   if (aFileMasq)
   {
      ELISE_COPY
      (
       rectangle(anOffset,anOffset+aMnt.sz()),
       aFileMasq->in(0) || trans(aMnt.in()>aSousDef,-anOffset),
       aFileMasq->out()
      );
   }

   delete aFileMasq;
}


void cEtapeMecComp::AllBasculeMnt(Pt2di aP0,Pt2di aP1,float ** aDataF,INT2 ** aDataI,Pt2di aSzData)
{
   std::list<cBasculeRes> aLBR = mEtape.BasculeRes();
   for 
   (
       std::list<cBasculeRes>::iterator itB = aLBR.begin();
       itB != aLBR.end();
       itB++
   )
   {
      OneBasculeMnt(aP0,aP1,*itB,aDataF,aDataI,aSzData);
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
