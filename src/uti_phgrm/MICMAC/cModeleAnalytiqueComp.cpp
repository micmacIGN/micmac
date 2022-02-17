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
// #include "XML_GEN/all.h"

/*
 
 ---------------------------------------------------------------------------------

 cMatrCorresp :  contient une correspondance "moyennee" entre deux images.

 Elle contient :

        mImX1, mImY1  :   P1 = mImX1(i,j),mImY1(i,j) est un point "moyenne" de l'image

        mImPds(i,j) est le poids associe a ce point

        mXY2 est la parallaxe moyennee de 
 
   Normalize  :

        va utiliser mImX1,mImY1,mImPds,mXY2  pour creer des points homologues.


	mPackHomInit  : "vrais" points homologues issus de l'appariements

	mPackHomCorr : point homologues corriges du maxim de modele a priori
	(ie si le modele etait parfait, on aurait P1=P2);

   cMatrCorresp * cModeleAnalytiqueComp::GetMatr(int aPas) cree une telle matrice 
   avec les deux "paquets" initialises.

void cModeleAnalytiqueComp::MakeExport()
{
   * cree le paquet avec GetMatr
     (eventuellement si UseHomologueReference est vrai, on lui subsititue a des
     fin de mise au point un paquet stocke sur fichier par ReadPackHom)
   * le filtre eventuellement avec FiltragePointHomologues (pour enlever les point
     hors image ou trop tres du bord);

   *  Ensuite on a un embranchement suivant que l'on est DHD ou en
      Ori :

          - DHD    SolveHomographie(*aPackIn);  // En general aPackIn vaut mPackHomCorr
                   SauvHom(*aPackIn);
          - Ori                           //     En general aPackIn vaut mPackHomCorr==mPackHomInit filtre
	          cMA_AffineOrient ...  (a Documenter + tard)
}

cModeleAnalytiqueComp en tant que Dist22Gen , sa fonction ::Direct est le modele
analytique calcule soit :

 1  Pt2dr aQ = mGeom2.CorrigeDist1(aP);   // Distorsion images 1
 2  aQ = CorrecDirecte(aQ);               // Modele complementaire (Hom+ ? Polynome)
 3  aQ = mGeom2.InvCorrDist2(aQ);         // Homographie de base + Distosion Image 2



void cModeleAnalytiqueComp::SolveHomographie(const ElPackHomologue &  aPackHom)
{
  1- estime un  modele analytique de la deformation aPackHom;

   ce modele est fait :

       d'une homographie (calculee par L1) composee eventuellement
       par un polynome de degre N  (!! Attention le polynome est aussi
       estime en L1 par Barrodale, Pb potentiel si degre eleve et 

    Le modele est modifie ce qui fait que lui 

  2- le mode est sauvegarde en XML, il sera reutilise a l'etape suivante (?
   a documenter .. avec  "LoadMA" qui va charger le modele);


  3- Sauvegarde Image si  (mModele.ExportImage().Val() || mModele.ReuseResiduelle().Val() )

      On calcule la paralaxe (a valeur reelle) qui aurait donne exactement la valeur
      du modele analytique


      Apparament (a eclaircir !) ReuseResiduelle n'a pas d'effet sur l'etape d'apres. BUG ??

  4- Sauvegarde du modele sous forme de grille "prete" a l'emploi


    if (mExpModeleGlobal) ...
}

*/



void  AssertAutomSelExportOriIsInit(const cOneModeleAnalytique & aM)
{
      ELISE_ASSERT
      (
         aM.AutomSelExportOri().IsInit(),
         "AutomNamesExportOri non init;  avec AutomSelExportOri init"
      );
}


/**********************************************************************/
/*                                                                    */
/*               cMC_PairIm                                           */
/*                                                                    */
/**********************************************************************/


cMC_PairIm::cMC_PairIm
(
     int aDZ,
     const cGeomDiscFPx & aGDF,
     cPriseDeVue & aPDV1,
     cPriseDeVue & aPDV2
) :
   mDZ      (aDZ),
   mResolZ1 (aGDF.ResolZ1()),
   mPDV1    (aPDV1),
   mPDV2    (aPDV2),
   mI1      (CreateAllImAndLoadCorrel<Im2D_REAL4>(mPDV1.IMIL(),aDZ)),
   mTI1     (mI1),
   mI2      (CreateAllImAndLoadCorrel<Im2D_REAL4>(mPDV2.IMIL(),aDZ)),
   mTI2     (mI2)
{
}

double cMC_PairIm::Correl(Pt2dr aPTER,int aSzW,double * aVPx)
{
    RMat_Inertie aMat;

    Pt2di aDP;
    for (aDP.x=-aSzW ; aDP.x<=aSzW; aDP.x++)
    {
        for (aDP.y=-aSzW ; aDP.y<=aSzW; aDP.y++)
        {
	    Pt2dr aQTer = aPTER +  (Pt2dr(aDP) * double(mDZ*mResolZ1));

	    Pt2dr aPI1 = mPDV1.Geom().Objet2ImageInit_Euclid(aQTer,aVPx) / double(mDZ);
	    Pt2dr aPI2 = mPDV2.Geom().Objet2ImageInit_Euclid(aQTer,aVPx) / double(mDZ);
	    if (mTI1.Rinside_bilin(aPI1) && (mTI2.Rinside_bilin(aPI2)))
	    {
	       aMat.add_pt_en_place(mTI1.getr(aPI1),mTI2.getr(aPI2));
	    }
	    else
	       return -1;
        }
    }

    return aMat.correlation();
}

/**********************************************************************/
/*                                                                    */
/*               cMatrCorresp                                         */
/*                                                                    */
/**********************************************************************/

cMatrCorresp::cMatrCorresp(cAppliMICMAC & anAppli,Pt2di aSz,int aNBXY2) :
   mAppli  (anAppli),
   mNBXY2 (aNBXY2),
   mSz    (aSz),
   mImPds (aSz.x,aSz.y,0.0),
   mTImPds (mImPds),
   mImX1  (aSz.x,aSz.y,0.0),
   mTImX1 (mImX1),
   mImY1  (aSz.x,aSz.y,0.0),
   mTImY1 (mImY1),
   mImZ   (1,1,0.0),
   mTImZ  (mImZ)
{
   for (int aK=0 ; aK<aNBXY2 ; aK++)
   {
     mXY2.push_back(Im2D_REAL4(aSz.x,aSz.y,0.0));
     mVYY2.push_back(mXY2.back().data());
   }
   mDXY2 = &(mVYY2[0]);
}

bool cMatrCorresp::IsOK(const Pt2di & aPRas) const
{
   return mTImPds.get(aPRas,0) > 0;
}

Pt2dr  cMatrCorresp::P1(const Pt2di & aPRas) const
{
   return Pt2dr
          (
              mTImX1.get(aPRas),
              mTImY1.get(aPRas)
          );
}

void cMatrCorresp::SetXY2(const Pt2di & aPRas,double * aVXY2) const
{
   for (int aK=0 ; aK<mNBXY2 ; aK++)
       aVXY2[aK] = mDXY2[aK][aPRas.y][aPRas.x];
}


//    Pt2dr P1 (const Pt2di & aPRas) const;
//    void SetXY2(const Pt2di &,double *) const;


const Pt2di & cMatrCorresp::Sz() const
{
   return mSz;
}

Output  cMatrCorresp::StdHisto()
{
   Output  aRes = Virgule(mImPds.histo(),mImX1.histo(),mImY1.histo());
   for (int aK=0 ; aK<int(mXY2.size()) ; aK++)
       aRes = Virgule(aRes,mXY2[aK].histo());

   return aRes;
}





void cMatrCorresp::Normalize
     (
          const cOneModeleAnalytique &      aModele,
          const cGeomDiscFPx &              aGT,
	  cPriseDeVue &               aPDV1,
          const cGeomImage &                aGeom1,
	  cPriseDeVue &               aPDV2,
          const cGeomImage &                aGeom2
     )
{
    const cReCalclCorrelMultiEchelle * aRCCME=GetOnlyUseIt(aModele.ReCalclCorrelMultiEchelle());

    bool isNuage3D = (aModele.TypeModele() == eTMA_Nuage3D);

    // A voir, sans doute un peu severe pour eTMA_Nuage3D
    if (isNuage3D)
    {
         mImZ = Im2D_REAL4(mSz.x,mSz.y,0.0);
	 mTImZ = TIm2D<REAL4,REAL8>(mImZ);
    }
    else
    {
        ELISE_ASSERT
        (
            aGeom1.IsId(),
            "Modele Analytique supose Geom1=Identite "
        );
    }

    double aVXY2[theDimPxMax];
    Pt2di aPRas;
    Im2D_REAL4 aIRCC(mSz.x,mSz.y,0.0);
    for (aPRas.x=0 ; aPRas.x<mSz.x ; aPRas.x++)
    {
        for (aPRas.y=0 ; aPRas.y<mSz.y ; aPRas.y++)
        {
           double aPds = mTImPds.get(aPRas);
           if ( aPds > 0)
           {
               Pt2dr aPTer
                     (
                         mTImX1.get(aPRas)/aPds,
                         mTImY1.get(aPRas)/aPds
                     );
               for (int aK=0; aK<mNBXY2; aK++)
               {
                   aVXY2[aK] = mDXY2[aK][aPRas.y][aPRas.x]/ aPds;
               }

               aGT.PxDisc2PxReel(aVXY2,aVXY2);
               aPTer = aGT.RDiscToR2(aPTer);

	       if (isNuage3D)
	       {
	             Pt3dr aPE = aGeom1.Restit2Euclid(aPTer,aVXY2);
                     mTImX1.oset(aPRas,aPE.x);
                     mTImY1.oset(aPRas,aPE.y);
                     mTImZ.oset(aPRas,aPE.z);
		     if (aPRas.y==0) 
		         std::cout << aPRas.x << "\n";
	       }
	       else
	       {
               
                   Pt2dr aQ1 =  aGeom1.Objet2ImageInit_Euclid(aPTer,aVXY2);
                   Pt2dr aQ2 =  aGeom2.Objet2ImageInit_Euclid(aPTer,aVXY2);


	           if (aRCCME!=0)
	           {
	               bool aMMin = aRCCME->AgregMin().Val();
	               double aCorAgr = aMMin ? 1e10 : 0;
		       int aNb=0;

		       for 
		       (
		           std::list<Pt2di>::const_iterator itScSz=aRCCME->ScaleSzW().begin();
		           itScSz!=aRCCME->ScaleSzW().end();
		           itScSz++
		       )
		       {
                           aNb++;
		           cMC_PairIm * aPair = mPairs[itScSz->x];
		           if (aPair==0)
		           {
	                       mPairs[itScSz->x] = new cMC_PairIm(itScSz->x,aGT,aPDV1,aPDV2);
		               aPair = mPairs[itScSz->x];
		           }
                           double aCor = aPair->Correl(aPTer,itScSz->y,aVXY2);
		           if (aMMin)
		              ElSetMin(aCorAgr,aCor);
                           else
		              aCorAgr += aCor;
		       }

                       if (! aMMin)
		          aCorAgr /= aNb;

                       aIRCC.data()[aPRas.y][aPRas.x] = (REAL4)aCorAgr;
                       // std::cout << "COR = " << aCorAgr << "\n";
		       if (aCorAgr < aRCCME->Seuil() )
		       {
                          aPds =  0;
		          mTImPds.oset(aPRas,0);
		       }
	           }

                   if (aPds > 0)
	           {
                      mPackHomInit.Cple_Add(ElCplePtsHomologues(aQ1,aQ2,aPds));

                      aQ1 =  aGeom2.CorrigeDist1(aQ1);
                      aQ2 =  aGeom2.CorrigeDist2(aQ2);
                      mPackHomCorr.Cple_Add(ElCplePtsHomologues(aQ1,aQ2,aPds));
                  }
	      }
           }
        }
    }


    if (aRCCME && (aRCCME->DoImg().Val()))
    {
        std::string aName=   mAppli.FullDirResult()
	                   + std::string("CorME_")
	                   + mAppli.NameChantier()
			   + std::string(".tif");
        Tiff_Im::Create8BFromFonc(aName,aIRCC.sz(),Max(0,Min(255,128*(1+aIRCC.in()))));

    }
}

const ElPackHomologue &   cMatrCorresp::PackHomCorr() const
{
   return mPackHomCorr;
}

const ElPackHomologue &   cMatrCorresp::PackHomInit() const
{
   return mPackHomInit;
}


Im2D_REAL4 cMatrCorresp::ImPds()  {return mImPds;}
Im2D_REAL4 cMatrCorresp::ImAppX() {return mImX1;}
Im2D_REAL4 cMatrCorresp::ImAppY() {return mImY1;}
Im2D_REAL4 cMatrCorresp::ImAppZ() {return mImZ;}
/*****************************************************************/
/*                                                               */
/*               cMA_AffineOrient                                */
/*                                                               */
/*****************************************************************/

 //'class cMA_AffineOrient' does not have a copy constructor
 //which is recommended since the class contains a pointer to allocated memory.
class cMA_AffineOrient
{
     public :
         cMA_AffineOrient
         (
               cAppliMICMAC &               anAppli,
               cModeleAnalytiqueComp        &,
               const cGeomDiscFPx &   mGeoTer,
               bool                        L1,
               const Pt2di &               aP,
               const ElPackHomologue &     aPackHom
         );
         void NoOp() {}
         void OneItere(bool isLast);
         void StateOri2(const std::string & aMes);

         void TestOri1Ori2(bool ShowAll,CamStenope &,CamStenope &);
         const ElPackHomologue & PackHom() const;
         Pt3dr  CalcPtMoy(CamStenope &,CamStenope &);

      // Calcul une image de px tranverse "ideale", c.a.d celle obenue
      // si la realite etait l'ori 
         void MakeImagePxRef();


    private :
       cAppliMICMAC &            mAppli;
       Pt2di                     mSzM;
       Im2D_REAL4                mImResidu;
       TIm2D<REAL4,REAL8>        mTImResidu;
       Im2D_REAL4                mImPds;
       TIm2D<REAL4,REAL8>        mTImPds;

       const cGeomDiscFPx &   mGeoTer;
       int              mPas;
       CamStenope *      mOri1;
       CamStenope *      mOri2;
       cOrientationConique  * mOC1;
       cOrientationConique  * mOC2;
       int              mNbPtMin;
       CamStenope &     mNewOri1;
       CamStenope &     mNewOri2;
       CamStenope &   mCam1;
       CamStenope &   mCam2;
       ElPackHomologue  mPack;
       cSetEqFormelles  mSetEq;
       ElRotation3D              mChR;
       cParamIntrinsequeFormel * mPIF;
       cCameraFormelle *         mCamF1;
       cCameraFormelle *         mCamF2;
       cCpleCamFormelle *        mCpl12;
       double                    mFocale;
};

const ElPackHomologue &   cMA_AffineOrient::PackHom() const
{
   return mPack;
}

Pt3dr cMA_AffineOrient::CalcPtMoy(CamStenope & anOri1,CamStenope & anOri2)
{
   Pt3dr aSPt(0,0,0);
   double aSPds =0.0;
   for 
   (
      ElPackHomologue::iterator iT = mPack.begin();
      iT != mPack.end();
      iT++
   )
   {
          double aD;
          Pt3dr aP = anOri1.PseudoInter(iT->P1(),anOri2,iT->P2(),&aD);
          aSPds += iT->Pds();
          aSPt =  aSPt + aP * iT->Pds();
    }

    return aSPt/aSPds;
}

void cMA_AffineOrient::TestOri1Ori2(bool ShowAll,CamStenope & anOri1,CamStenope & anOri2)
{
   double aSEc2 =0;
   double aSEc =0;
   double aSP =0;
   for 
   (
      ElPackHomologue::iterator iT = mPack.begin();
      iT != mPack.end();
      iT++
   )
   {
          Pt2dr aQ1  = mCam1.F2toPtDirRayonL3(iT->P1());
          Pt2dr aQ2  = mCam2.F2toPtDirRayonL3(iT->P2());
          double aL =  mCpl12->ResiduSigneP1P2(aQ1,aQ2);
          aL = ElAbs(aL*mFocale);


          double aD;
          anOri1.PseudoInter(iT->P1(),anOri2,iT->P2(),&aD);
          aSEc2 += ElSquare(aD)*iT->Pds();
          aSEc  += ElAbs(aD)*iT->Pds();
          aSP   += iT->Pds();

       // double aRatio = aL/aD;
       // Pt2dr  aP1 = mGeoTer.R2ToRDisc(iT->P1());
       // Pt2di  aI1  =  round_down(aP1/mPas);
/*
          {
             if (ShowAll) std::cout  << aL << aP << "\n";
                   << iT->Pds() << " "
                   << iT->P1()  << " "
                   << iT->P2()  << " "
                   << " RATIO = " << aL/aD  << " " 
                   << " Ec Ang " <<  aL     << " "
                   << " D INTER = " << aD << "\n";
          }
*/
   }
   std::cout << "EC2 = " << sqrt(aSEc2/aSP) 
             <<  " ECAbs = " << (aSEc/aSP) << "\n";

   for 
   (
        std::list<cListTestCpleHomol>::iterator itC=mAppli.ListTestCpleHomol().begin();
        itC !=mAppli.ListTestCpleHomol().end();
        itC++
   )
   {
       // Pt2dr aP1 = itC->PtIm1();
       // Pt2dr aP2 = itC->PtIm2();
       double aD;
       Pt3dr aP = anOri1.PseudoInter(itC->PtIm1(),anOri2,itC->PtIm2(),&aD);
 
       std::cout << "VERIF " << aP << " " << aD << "\n";
   }
}

void cMA_AffineOrient::StateOri2(const std::string & aMes)
{
   // OO Pt3dr aSom = mOri2->orsommet_de_pdv_terrain();
   Pt3dr aSom = mOri2->PseudoOpticalCenter();
   Pt3dr aTr  = mCamF2->RF().CurRot().tr();
   std::cout  << aMes << (aSom-aTr) << " "
             << aSom << " " << aTr << "\n";
}

void cMA_AffineOrient::OneItere(bool isLast)
{
   ElRotation3D aR1 = mCamF1->RF().CurRot();
   ElRotation3D aR2 = mCamF2->RF().CurRot();

      std::cout << "TR = " << (aR2.tr()-aR1.tr())  << "\n";
      std::cout << "I = " <<   aR2.ImVect(Pt3dr(1,0,0))  << euclid(aR2.ImVect(Pt3dr(1,0,0))) << "\n";
      std::cout << "J = " <<   aR2.ImVect(Pt3dr(0,1,0))  << euclid(aR2.ImVect(Pt3dr(0,1,0))) << "\n";
      std::cout << "K = " <<   aR2.ImVect(Pt3dr(0,0,1))  << euclid(aR2.ImVect(Pt3dr(0,0,1))) << "\n";

   mSetEq.AddContrainte(mPIF->StdContraintes(),true);
   mSetEq.AddContrainte(mCamF1->RF().StdContraintes(),true);
   mSetEq.AddContrainte(mCamF2->RF().StdContraintes(),true);

   double aSEc2 =0;
   double aSEc =0;
   double aSP =0;
   int aNbP=0;
   for 
   (
      ElPackHomologue::iterator iT = mPack.begin();
      iT != mPack.end();
      iT++
   )
   {
       // double aD =  mCam1.EcartProj(iT->P1(),mCam2,iT->P2());
       Pt2dr aQ1  = mCam1.F2toPtDirRayonL3(iT->P1());
       Pt2dr aQ2  = mCam2.F2toPtDirRayonL3(iT->P2());
       //double aL =  mCpl12->ResiduSigneP1P2(aQ1,aQ2);
       double aL2 =  mCpl12->AddLiaisonP1P2(aQ1,aQ2,iT->Pds(),false);
       aSEc2 += ElSquare(aL2)*iT->Pds();
       aSEc += ElAbs(aL2)*iT->Pds();
       aSP  += iT->Pds();

       Pt2dr  aP1 = mGeoTer.R2ToRDisc(iT->P1());
       Pt2di  aI1  =  round_down(aP1/mPas);
       mTImResidu.oset(aI1,aL2*mFocale);
       mTImPds.oset(aI1,iT->Pds());

       if (iT->Pds() > 1e-5)
          aNbP++;
   }

   if (aNbP < mNbPtMin)
   {
      mAppli.MicMacErreur
      (
         eErrNbPointInEqOriRel,
         "Pas assez de point pour equation d'orientation relative",
         "Zone de recouvrement trop faible"
      );
   }


   mSetEq.SolveResetUpdate();
   // if (isLast)
   {
       std::cout 
           << aSEc2 << " " << aSP << " " << mPack.size()
           << " EQuad = " << (sqrt(aSEc2/aSP)*mFocale) 
           << " EAbs  = " << ((aSEc/aSP)*mFocale) 
           << "\n";
  } 
}


void cMA_AffineOrient::MakeImagePxRef()
{
    int aSsRes = 20;
    Pt2di aSzGlob = mAppli.SzOfResol(1);

    Pt2di aPRes1;
    Pt2di aPInd;
    cGeomDiscFPx aGDF = mAppli.GeomDFPxInit();

    const cGeomImage & aGeom2 = mAppli.PDV2()->Geom();
    double aZ0 = mNewOri2.GetAltiSol();

    Pt2di aSzSR = (aSzGlob+Pt2di(aSsRes-1,aSsRes-1))/aSsRes;
    Im2D_REAL4 mImRes(aSzSR.x,aSzSR.y,0.0);
    TIm2D<REAL4,REAL8> mTImRes(mImRes);
    

    for (aPRes1.x=aPInd.x=0 ; aPRes1.x<aSzGlob.x ; aPRes1.x+=aSsRes,aPInd.x++)
    {
        for (aPRes1.y=aPInd.y=0 ; aPRes1.y<aSzGlob.y ; aPRes1.y+=aSsRes,aPInd.y++)
        {
            Pt2dr aPIm1 = aGDF.DiscToR2(aPRes1);
            Pt3dr aPTer  =  mNewOri1.ImEtZ2Terrain(aPIm1,aZ0);
            Pt2dr aPIm2   = mNewOri2.R3toF2(aPTer);
            Pt2dr aPx = aGeom2.P1P2ToPx(aPIm1,aPIm2);
            mTImRes.oset(aPInd,aPx.y);
            // std::cout << aPx.y << "\n";
// (aP2Ter.x,aP2Ter.y,mNewOri2.altisol());
            // Pt2dr  aPIm2 = mNewOri2.to_photo(aP3Ter);
        }
    }
    Tiff_Im::Create8BFromFonc
    (
           "toto.tif",
           aSzGlob,
            Max(0,Min(255,120+100.0*mImRes.in(0)[Virgule(FX,FY)/double(aSsRes)]))
    );
}

static tParamAFocal aNoPAF;

cMA_AffineOrient::cMA_AffineOrient
(
    cAppliMICMAC &               anAppli,
    cModeleAnalytiqueComp & aModele,
    const cGeomDiscFPx &    aGeoTer,
    bool                    L1,
    const Pt2di &               aSzM,
    const ElPackHomologue &     aPackHom
)  :
   mAppli     (anAppli),
   mSzM       (aSzM),
   mImResidu  (mSzM.x,mSzM.y),
   mTImResidu (mImResidu),
   mImPds     (mSzM.x,mSzM.y),
   mTImPds    (mImPds),
   mGeoTer (aGeoTer),
   mPas    (aModele.Modele().PasCalcul()),
   mOri1   (aModele.mGeom1.GetOriNN()),
   mOri2   (aModele.mGeom2.GetOriNN()),
   mOC1    (new cOrientationConique(mOri1->StdExportCalibGlob())),
   mOC2    (new cOrientationConique(mOri2->StdExportCalibGlob())),
   // OO mOC1    (mOri1->OC() ? new cOrientationConique(*(mOri1->OC())) : 0),
   // OO mOC2    (mOri2->OC() ? new cOrientationConique(*(mOri2->OC())) : 0),
   mNbPtMin  (aModele.Modele().NbPtMinValideEqOriRel().Val()),
   mNewOri1 (*(mOri1->Dupl())),
   mNewOri2 (*(mOri2->Dupl())),
   mCam1   (*mOri1),
   mCam2   (*mOri2),
   mPack   (aPackHom),
   mSetEq  (
              (L1 ? cNameSpaceEqF::eSysL1Barrodale : cNameSpaceEqF::eSysPlein),
              mPack.size()
           ),
   mChR    (Pt3dr(0,0,0),0,0,0),
   // mSetEq (cNameSpaceEqF::eSysL1Barrodale,mPack.size()),

             // On va donner des points corriges de  la dist :
   // mPIF    (mSetEq.NewParamIntrNoDist(1.0,Pt2dr(0,0))),
   mPIF    (mSetEq.NewParamIntrNoDist(true,new CamStenopeIdeale(true,1.0,Pt2dr(0,0),aNoPAF))),

   
   mCamF1  (mPIF->NewCam
                 (
                      cNameSpaceEqF::eRotFigee,
                      mChR * mCam1.Orient().inv()
                 )
            ),
    mCamF2  (mPIF->NewCam
                 (
                      cNameSpaceEqF::eRotBaseU,
                      mChR * mCam2.Orient().inv(),
                      mCamF1
                 )
             ),
    mCpl12   (mSetEq.NewCpleCam(*mCamF1,*mCamF2)),
    mFocale  (mOri1->Focale())
{


   mSetEq.SetClosed();
   mPIF->SetFocFree(false);
   mPIF->SetPPFree(false);

   int aNbEtape = 9;
   int aFlag = 0;
   for 
   (
      std::list<int>::const_iterator itB = aModele.Modele().NumsAngleFiges().begin();
      itB != aModele.Modele().NumsAngleFiges().end();
      itB++
   )
      aFlag  |= 1 << *itB;


   for (int aK=0 ; aK<aNbEtape ; aK++)
   {
      mCamF2->RF().SetFlagAnglFige(aFlag);
/*
       mCamF2->RF().SetModeRot
       (
            (aK<3) ?
            cNameSpaceEqF::eRotCOptFige :
            cNameSpaceEqF::eRotBaseU
       );
*/
       OneItere(aK==(aNbEtape-1)); 
   }
   std::string  aName =    mAppli.FullDirResult() 
                         + std::string("Residus_Dz")
                         +  ToString(round_ni(mGeoTer.DeZoom()))
                         + std::string("_")
                         + mAppli.NameChantier() 
                         + std::string(".tif");
    Tiff_Im aFileResidu
            (
                aName.c_str(),
                mSzM,
                GenIm::u_int1,
                Tiff_Im::No_Compr,
                Tiff_Im::RGB
            );


   mNewOri2.SetOrientation(mCamF2->RF().CurRot());
   Pt3dr  aPMoy = CalcPtMoy(mNewOri1,mNewOri2);
   mNewOri1.SetAltiSol(aPMoy.z);
   mNewOri2.SetAltiSol(aPMoy.z);

   
   std::string aNAu = aModele.Modele().AutomSelExportOri().Val();
   std::string aNEx1 = aModele.Modele().AutomNamesExportOri1().Val();
   std::string aNEx2 = aModele.Modele().AutomNamesExportOri2().Val();
   std::string aNI1 = mAppli.PDV1()->Name();
   std::string aNI2 = mAppli.PDV2()->Name();
   std::string aNOri1 =   mAppli.FullDirGeom()
                        + StdNameFromCple(aModele.AutomExport(),aNAu,aNEx1,"@",aNI1,aNI2);
   std::string aNOri2 =   mAppli.FullDirGeom()
                        + StdNameFromCple(aModele.AutomExport(),aNAu,aNEx2,"@",aNI1,aNI2);

   bool aXmlRes = false;

   if (StdPostfix(aNOri1)== "xml")
   {
       aXmlRes = true;
       ELISE_ASSERT
       (
              (StdPostfix(aNOri2)=="xml")
	   && (mOC1!=0) && (mOC2!=0),
	   "Incoherence in  XML export for cMA_AffineOrient"
       );
       // Les points de verifs, si ils existent n'ont pas de raison d'etre transposables
       mOC1->Verif().SetNoInit();
       mOC2->Verif().SetNoInit();

       ElRotation3D aR1 = mCamF1->RF().CurRot();
       ElRotation3D aR2 = mCamF2->RF().CurRot();
       mOC2->Externe()  = From_Std_RAff_C2M(aR2,mOC2->Externe().ParamRotation().CodageMatr().IsInit());

       mOC1->Externe().AltiSol().SetVal(aPMoy.z);
       mOC2->Externe().AltiSol().SetVal(aPMoy.z);

       mOC1->Externe().Profondeur().SetVal(ProfFromCam(aR1.inv(),aPMoy));
       mOC2->Externe().Profondeur().SetVal(ProfFromCam(aR2.inv(),aPMoy));

   }

   TestOri1Ori2(true,*mOri1,*mOri2);
   TestOri1Ori2(true,mNewOri1,mNewOri2);
   std::cout << "ZMoyen = " << aPMoy.z  << "\n";

   Fonc_Num aFoK = (mImPds.in()>0);
   Fonc_Num aFRes = Max(0,Min(255,128.0 +mImResidu.in()*20));
   ELISE_COPY
   (
        aFileResidu.all_pts(),
          aFoK*its_to_rgb(Virgule(aFRes,3.14,32*Abs(mImResidu.in()>0.5)))
        + (1-aFoK)*Virgule(255,0,0),
        aFileResidu.out()
   );

   bool Exp1 =  aModele.Modele().AutomNamesExportOri1().IsInit();
   bool Exp2 =  aModele.Modele().AutomNamesExportOri2().IsInit();
   ELISE_ASSERT(Exp1 == Exp2,"Incoherence in AutomNamesExportOri");
   if (Exp1)
   {
      std::cout << "EXPORT   OOOOOOOOOOOOOOORI\n";
      AssertAutomSelExportOriIsInit(aModele.Modele());

      if(aXmlRes)
      {
          MakeFileXML(*mOC1,aNOri1,"MicMacForAPERO");
          MakeFileXML(*mOC2,aNOri2,"MicMacForAPERO");
      }
      else
      {
          ELISE_ASSERT(false,"Cannot write ori ");
          // OO mNewOri1.write_txt(aNOri1.c_str());
          // OO mNewOri2.write_txt(aNOri2.c_str());
      }
   }


   
   if (aModele.Modele().SigmaPixPdsExport().ValWithDef(-1) > 0)
   {
      double aPds = aModele.Modele().SigmaPixPdsExport().Val();
      for 
      (
         ElPackHomologue::iterator iT = mPack.begin();
         iT != mPack.end();
         iT++
      )
      {
          Pt2dr aQ1  = mCam1.F2toPtDirRayonL3(iT->P1());
          Pt2dr aQ2  = mCam2.F2toPtDirRayonL3(iT->P2());
          double aL =  ElAbs(mCpl12->ResiduSigneP1P2(aQ1,aQ2))*mFocale;
          double aP = ElSquare(aPds)/(ElSquare(aPds)+ElSquare(aL));
          iT->Pds() *= aP;
      }
   }
}


/**********************************************************************/
/*                                                                    */
/*               cModeleAnalytiqueComp                                */
/*                                                                    */
/**********************************************************************/

cModeleAnalytiqueComp::cModeleAnalytiqueComp
(
  cAppliMICMAC &               anAppli,
  const cOneModeleAnalytique & aModele,
  cEtapeMecComp &              anEtape
) :
   mSz         (-1,-1),
   mAppli      (anAppli),
   mModele     (aModele),
   mEtape      (&anEtape),
   mNbPx       (anAppli.DimPx()),
   mPDV1       (*mAppli.PDV1()),
   mGeom1      (mPDV1.Geom()),
   mPDV2       (*mAppli.PDV2()),
   mGeom2      (mPDV2.Geom()),
   mGeoTer     (mEtape->GeomTer()),
   mHomogr     (cElHomographie::Id()),
   mHomogrInv  (cElHomographie::Id()),
   mPolX       (1,1.0),
   mPolY       (1,1.0),
   mDegrPolAdd (mModele.DegrePol().ValWithDef(-1)),
   mPolXInv    (1,1.0),
   mPolYInv    (1,1.0),
   mNameXML    (NameFile(true,"Homographie","xml")),
   mNameImX    (NameFile(true,"ImageX","tif")),
   mNameImY    (NameFile(true,"ImageY","tif")),
   mNameResX   (NameFile(true,"ResiduX","tif")),
   mNameResY   (NameFile(true,"ResiduY","tif")),
   mExpModeleGlobal (   mModele.FCND_ExportModeleGlobal().IsInit()),
   mAutomExport     (0)
{
   mGeoTer.SetClip(Pt2di(0,0),mGeoTer.SzDz());

}

bool cModeleAnalytiqueComp::ExportGlob() const
{
   return mExpModeleGlobal;
}

void cModeleAnalytiqueComp::MakeInverseModele()
{
   mHomogrInv = mHomogr.Inverse();
   if (mDegrPolAdd>0)
   {
      ElDistortionPolynomiale aDPol(mPolX,mPolY);
      
      ElDistortionPolynomiale aDPolInv = aDPol.NewPolynLeastSquareInverse
                                         (
                                              mBoxPoly,
                                              mPolX.DMax() + 2
                                         );
       mPolXInv = aDPolInv.DistX();
       mPolYInv = aDPolInv.DistY();

   }
}

void cModeleAnalytiqueComp::LoadMA()
{
   cElXMLTree aTree(mNameXML);
   mHomogr = aTree.GetUnique("Homographie")->GetElHomographie();
   if (mDegrPolAdd>0)
   {
      mPolX = aTree.GetUnique("XPoly")->GetPolynome2D();
      mPolY = aTree.GetUnique("YPoly")->GetPolynome2D();

      mBoxPoly._p0 = aTree.GetUnique("PMinBox")->GetPt2dr();
      mBoxPoly._p1 = aTree.GetUnique("PMaxBox")->GetPt2dr();
   }
   MakeInverseModele();
}

Pt2dr cModeleAnalytiqueComp::CorrecDirecte(const Pt2dr & aP) const
{
   Pt2dr aQ = mHomogr.Direct(aP);
   if (mDegrPolAdd>0)
   {
       aQ = Pt2dr(mPolX(aQ),mPolY(aQ));
   }
   return aQ;
}
Pt2dr cModeleAnalytiqueComp::CorrecInverse(const Pt2dr & aP) const
{
   Pt2dr aQ = aP;
   if (mDegrPolAdd>0)
   {
       aQ = Pt2dr(mPolXInv(aQ),mPolYInv(aQ));
   }
   return mHomogrInv.Direct(aQ);
}


Pt2dr cModeleAnalytiqueComp::Direct(Pt2dr aP) const
{
   Pt2dr aQ = mGeom2.CorrigeDist1(aP);
   aQ = CorrecDirecte(aQ);
   aQ = mGeom2.InvCorrDist2(aQ);
   return  aQ;
}

bool cModeleAnalytiqueComp::OwnInverse(Pt2dr & aP) const 
{
   aP = mGeom2.CorrigeDist2(aP);
   aP = CorrecInverse(aP);
   aP = mGeom2.InvCorrDist1(aP);
   return true;
}



std::string cModeleAnalytiqueComp::NameFile
                    (
                         bool  aDirMEc,
                         const std::string & aName,
                         const std::string & aPost
                    )
{
   return    (aDirMEc ?  mAppli.FullDirMEC() : mAppli.FullDirGeom())
           + mModele.NameExport().Val()
           + std::string("_")
           + mAppli.NameChantier()
           + std::string("_")
           + aName
           + std::string("_Num")
           + ToString(mEtape->Num())
           + std::string(".")
           + aPost;
}

static Fonc_Num ToUC(Fonc_Num aF)
{
  return Max(0,Min(255,128+aF));
}

Fonc_Num  cModeleAnalytiqueComp::ImPx(int aK)
{
   return mEtape->KPx(aK).FileIm().in();
}


void cModeleAnalytiqueComp::SolveHomographie(const ElPackHomologue &  aPackHom)
{
   mHomogr = cElHomographie(aPackHom,mModele.HomographieL2().Val());

   mBoxPoly = Box2dr(Pt2dr(0,0),Pt2dr(0,0));
   if (mDegrPolAdd >0)
   {
      ElPackHomologue aPck2 = aPackHom;
      Pt2dr aPMin(1e5,1e5);
      Pt2dr aPMax(-1e5,-1e5);
      for 
      (
         ElPackHomologue::iterator iT = aPck2.begin();
         iT != aPck2.end();
         iT++
      )
      {
           iT->P1() = mHomogr.Direct(iT->P1());
           aPMin.SetInf(iT->P1());
           aPMax.SetSup(iT->P1());
      }
      double anAmpl = ElMax(dist8(aPMin),dist8(aPMax));
      mBoxPoly = Box2dr(aPMin,aPMax);
      bool aPL2 = mModele.PolynomeL2().Val();
      mPolX = aPck2.FitPolynome(aPL2,mDegrPolAdd,anAmpl,true);
      mPolY = aPck2.FitPolynome(aPL2,mDegrPolAdd,anAmpl,false);

   }
// Polynome2dReal  ElPackHomologue::FitPolynome

   {
      cElXMLFileIn aFileXML(mNameXML);
      aFileXML.PutElHomographie(mHomogr,"Homographie");

      if (mDegrPolAdd >0)
      {
         cElXMLFileIn::cTag aTag(aFileXML,"PolynomeCompl"); aTag.NoOp();
         
         aFileXML.PutPoly(mPolX,"XPoly");
         aFileXML.PutPoly(mPolY,"YPoly");
         
         aFileXML.PutPt2dr(mBoxPoly._p0,"PMinBox");
         aFileXML.PutPt2dr(mBoxPoly._p1,"PMaxBox");
      }
   } 
   MakeInverseModele();

   // Verification de la correction du calcul  de l'inverse
    if (0)
   {
      int aNb=10;
      double aEps = 0.05;
      for (int aKx=0 ; aKx<= aNb ; aKx++)
      {
          double aPdsX = ElMax(aEps,ElMin(1-aEps,aKx/double(aNb)));
          for (int aKy=0 ; aKy<= aNb ; aKy++)
          {
              
              double aPdsY = ElMax(aEps,ElMin(1-aEps,aKy/double(aNb)));
              Pt2dr aPRas = Pt2dr (mSzGl.x*aPdsX, mSzGl.x*aPdsY);
              Pt2dr aPTer = mGeoTer.RDiscToR2(aPRas);
              Pt2dr aP1 = Direct(Pt2dr(aPTer));

              Pt2dr aQ2 =  Inverse(aP1);
              double anEr = euclid(aPTer,aQ2);
              if (anEr>0.05)
              {
                 std::cout << "Erreur = " << anEr << "\n";
                 std::cout <<  aPRas <<  aPTer << "\n";
                 
                 Pt2dr aP0 = aPTer;
                 Pt2dr aP1 = mGeom2.CorrigeDist1(aP0);
                 Pt2dr aP2 = CorrecDirecte(aP1);
                 Pt2dr aP3 = mGeom2.InvCorrDist2(aP2);

                 Pt2dr aQ0 = mGeom2.InvCorrDist1(aP1);
                 std::cout << "pq0= " << euclid(aP0,aQ0) << aP0 << aQ0 << "\n";

                 Pt2dr aQ1 = CorrecInverse(aP2);
                 std::cout << "pq1= " << euclid(aP1,aQ1) << aP1 << aQ1 << "\n";

                 Pt2dr aQ2 =  mGeom2.CorrigeDist2(aP3);
                 std::cout << "pq2= " << euclid(aP2,aQ2) << aP2 << aQ2 << "\n";
                 std::cout << "P3 " << aP3 << "\n";

                  ELISE_ASSERT(false,"MakeInverseModele Pb!!");
              }
         }
     }
   }

   if (mModele.ExportImage().Val() || mModele.ReuseResiduelle().Val() )
   {
      Im2D_REAL4 anImX(mSzGl.x,mSzGl.y);
      TIm2D<REAL4,REAL8> aTImX(anImX);
      Im2D_REAL4 anImY(mSzGl.x,mSzGl.y);
      TIm2D<REAL4,REAL8> aTImY(anImY);

      Pt2di aPRas;
      double aPX0[2] = {0.0,0.0};
      for (aPRas.x=0 ; aPRas.x<mSzGl.x ; aPRas.x++)
      {
          for (aPRas.y=0 ; aPRas.y<mSzGl.y ; aPRas.y++)
          {
              Pt2dr aPTer = mGeoTer.DiscToR2(aPRas);
              Pt2dr aP1 = Direct(Pt2dr(aPTer));


              Pt2dr aP2 ;
              if (mModele.ReuseResiduelle().Val())
                 aP2 = mGeom2.Objet2ImageInit_Euclid(Pt2dr(aPTer),aPX0);
              else
                 aP2 = mGeom2.InvCorrDist2(mGeom2.CorrigeDist1(aPTer));
              double aPx[2];
              aPx[0] = aP1.x- aP2.x;
              aPx[1] = aP1.y- aP2.y;
              mGeoTer.PxReel2PxDisc(aPx,aPx);
              
               
              aTImX.oset(aPRas,aPx[0]);
              aTImY.oset(aPRas,aPx[1]);
          }
      }
      if (mModele.ExportImage().Val())
      {
         Tiff_Im::Create8BFromFonc(mNameImX,mSzGl,ToUC(anImX.in()));
         Tiff_Im::Create8BFromFonc(mNameImY,mSzGl,ToUC(anImY.in()));
      }
      if (mModele.ReuseResiduelle().Val())
      {
         Tiff_Im::Create8BFromFonc(mNameResX,mSzGl,ToUC(anImX.in()-ImPx(0)));
         Tiff_Im::Create8BFromFonc(mNameResY,mSzGl,ToUC(anImY.in()-ImPx(1)));
      }
  } 
  if (mExpModeleGlobal)
  {

      const cGeomDiscFPx &  aG = mAppli.GeomDFPxInit() ;
      Box2dr aBox = aG.BoxEngl();
      double aME =mModele.MailleExport().Val();


      cDbleGrid  aGrid
                 (
                     true, // P0P1 Direct par defaut maintien du comp actuel
                     true,
                     aBox._p0,aBox._p1,
                     Pt2dr(aME,aME),
                     *this,
                     "toto"
                 );


     std::string aNameXML =  mAppli.ICNM()->Assoc1To2
                             (
			        mModele.FCND_ExportModeleGlobal().Val(),
                                mAppli.PDV1()->Name(),
                                mAppli.PDV2()->Name(),
				true
			     );
     aGrid.SaveXML(mAppli.FullDirResult() + aNameXML);
  }

}


cMatrCorresp * cModeleAnalytiqueComp::GetMatr(int aPas,bool PointUnique)
{
    int aDz = mEtape->DeZoomTer();
    mSzGl = mAppli.SzOfResol(aDz);
    mSz = (mSzGl+Pt2di(aPas-1,aPas-1))/aPas;

    cMatrCorresp * pMatr = new cMatrCorresp (mAppli,mSz,mNbPx);

    Fonc_Num aFPds = mAppli.FoncMasqOfResol(aDz);
    if (mAppli.OneDefCorAllPxDefCor().Val())
       aFPds = aFPds &&  mEtape->FileMasqOfNoDef().in();

    if (PointUnique)
    {
          aFPds = aFPds * ((FX%aPas)==(aPas/2)) * ((FY%aPas)==(aPas/2));
    }

    // std::cout << "AAAAAAAAAa\n";
    if (mModele.FiltreByCorrel().Val())
    {
          Im1D_REAL8 aLut(256);
          double aS = mModele.SeuilFiltreCorrel().Val();
          double anExp = mModele.ExposantPondereCorrel().Val();
	  Fonc_Num FC = FX/128.0-1;
	  ELISE_COPY
	  (
	       aLut.all_pts(),
	       (mModele.UseFCBySeuil().Val()) ? 
	       (FC>=aS) : 
	       pow(Max(0.0,FC-aS),anExp),
	       aLut.out()
	  );
	  aFPds = aFPds * aLut.in()[mEtape->LastFileCorrelOK().in()];
	  /*
          double aS = mModele.SeuilFiltreCorrel().Val();
          Fonc_Num FC = mEtape->LastFileCorrelOK().in()/128.0-1;
          if (mModele.UseFCBySeuil().Val())
             aFPds = aFPds * (FC>=aS);
          else
             aFPds = aFPds * Max(0.0,(FC-aS));
	  */
    }

    Symb_FNum fM(aFPds);

/*
{
   double aSom;
    ELISE_COPY
    (
          rectangle(Pt2di(0,0),mSzGl),
          fM,
          sigma(aSom)
    );
    std::cout << "SssDSOM = " << aSom << "\n";
    getchar();
}
*/


    Fonc_Num  aFonc = Virgule(fM,fM*FX,fM*FY);

    for (int aK=0; aK<mNbPx ; aK++)
    {
       aFonc = Virgule(aFonc,fM*ImPx(aK));
    }

    ELISE_COPY
    (
          rectangle(Pt2di(0,0),mSzGl),
          aFonc,
          pMatr->StdHisto().chc(Virgule(FX,FY)/aPas)
    );
    pMatr->Normalize(mModele,mGeoTer,mPDV1,mGeom1,mPDV2,mGeom2);

    return pMatr;
}


void  cModeleAnalytiqueComp::TifSauvHomologues(const ElPackHomologue & aPack)
{
  int aPas = mModele.PasCalcul();
  if (! mModele.AutomNamesExportHomTif().IsInit())
     return;

  ELISE_ASSERT
  (
     mAppli.ExportForMultiplePointsHomologues().Val(),
     "TifSauvHom, No::ExportForMultiplePointsHomologues"
  ); 

  if (mAppli.EchantillonagePtsInterets().IsInit())
  {
     ELISE_ASSERT
     (
        mAppli.EchantillonagePtsInterets().IsInit(),
        "TifSauvHom, No::EchantillonagePtsInterets"
     ); 

     ELISE_ASSERT
     (
         mAppli.EchantillonagePtsInterets().Val().FreqEchantPtsI()==aPas,
         "TifSauvHom, PasCalcul!=FreqEchantPtsI"
     );
  }

  Pt2di aSzR = (mAppli.PDV1()->SzIm() + Pt2di(aPas-1,aPas-1)) / aPas;
  cElImPackHom aImPack(aPack,aPas,aSzR);




   std::string aNameTif = StdNameFromCple
                          (
                                  mAutomExport,
                                  mModele.AutomSelExportOri().Val(),
                                  mModele.AutomNamesExportHomTif().Val(),
                                  "@",
                                  mAppli.PDV1()->Name(),
                                  mAppli.PDV2()->Name()
                              );

   aImPack.SauvFile(mAppli.FullDirResult()+aNameTif);
}



// cDbleGrid

void cModeleAnalytiqueComp::SauvHomologues(const ElPackHomologue & aPack)
{


   TifSauvHomologues(aPack);

   if (mModele.AutomNamesExportHomXml().IsInit())
   {
       AssertAutomSelExportOriIsInit(mModele);
       std::string aNameXML = StdNameFromCple
                              (
                                  mAutomExport,
                                  mModele.AutomSelExportOri().Val(),
                                  mModele.AutomNamesExportHomXml().Val(),
                                  "@",
                                  mAppli.PDV1()->Name(),
                                  mAppli.PDV2()->Name()
                              );
       cElXMLFileIn aFileXML(mAppli.FullDirResult()+aNameXML);
       aFileXML.PutPackHom(aPack);
   }
   if (mModele.KeyNamesExportHomXml().IsInit())
   {
         std::string aNameXML = mAppli.ICNM()->Assoc1To2
	                        (
				    mModele.KeyNamesExportHomXml().Val(),
                                    mAppli.PDV1()->Name(),
                                    mAppli.PDV2()->Name(),
				    true
				);
         // cElXMLFileIn aFileXML(mAppli.WorkDir()+aNameXML);
         // aFileXML.PutPackHom(aPack);
         aPack.StdPutInFile(mAppli.WorkDir()+aNameXML);
   }
   if (mModele.AutomNamesExportHomBin().IsInit())
   {
       AssertAutomSelExportOriIsInit(mModele);
       std::string aNameBin = mAppli.FullDirResult()
                            + StdNameFromCple
                              (
                                  mAutomExport,
                                  mModele.AutomSelExportOri().Val(),
                                  mModele.AutomNamesExportHomBin().Val(),
                                  "@",
                                  mAppli.PDV1()->Name(),
                                  mAppli.PDV2()->Name()
                              );
        ELISE_fp aFP (aNameBin.c_str(),ELISE_fp::WRITE);
        aPack.write(aFP);
        aFP.close();
   }
}

bool  cModeleAnalytiqueComp::FiltragePointHomologues
      (
          const ElPackHomologue & aPackInit,
	  ElPackHomologue & aNewPack, 
	  double aTol,
	  double aFiltre
      )
{
       bool GotOut = false;
       Box2dr aBox1 = mAppli.PDV1()->BoxIm().AddTol(aTol);
       Box2dr aBox2 = mAppli.PDV2()->BoxIm().AddTol(aTol);

       Box2dr aBoxF1 = mAppli.PDV1()->BoxIm().AddTol(aFiltre);
       Box2dr aBoxF2 = mAppli.PDV2()->BoxIm().AddTol(aFiltre);

       for 
       (
          ElPackHomologue::const_iterator iT = aPackInit.begin();
          iT != aPackInit.end();
          iT++
       )
       {
          if (    ( aBoxF1.inside(iT->P1()))
	       && ( aBoxF2.inside(iT->P2()))
             )
          {

              bool thisOut = (! aBox1.inside(iT->P1()))
                          || (! aBox2.inside(iT->P2()));
              if (thisOut)
              {
                  std::cout << aBox1._p1 << aBox2._p1 << "\n";
                  std::cout << "OUT " << iT->P1() << " " << iT->P2() << " " << iT->Pds()<< "\n";
              }
              GotOut =    GotOut || thisOut;

	     aNewPack.Cple_Add(iT->ToCple());
          }
       }
       return GotOut;
}

void cModeleAnalytiqueComp::MakeExport()
{
   if (!mModele.MakeExport().Val())
      return;

    cMatrCorresp * pMatr = GetMatr(mModele.PasCalcul(),mModele.PointUnique().Val());

    const ElPackHomologue *  aPackIn =0;
    ElPackHomologue  aPackRef;
    ElPackHomologue  aNewPack;
    if (mModele.UseHomologueReference().Val())
    {
        aPackRef =  mAppli.PDV1()->ReadPackHom(mAppli.PDV2());
        aPackIn = & aPackRef;
    }
    else
    {
        aPackIn = & pMatr->PackHomCorr();
    }
 

    /* 
     *      FILTRAGE EVENTUEL DES POINTS HOMOLOGUES 
    */

    double aTol = mAppli.TolerancePointHomInImage().Val();
    double aFiltre = mAppli.FiltragePointHomInImage().Val();

    bool GotOut = false;

   
    switch (mModele.TypeModele())
    {

         case eTMA_Homologues :
              SauvHomologues(pMatr->PackHomInit());
         break;
         
         case eTMA_DHomD :
	 // std::cout << "PKS = " << aPackIn->size() << "\n"; getchar();
              SolveHomographie(*aPackIn);
              SauvHomologues(pMatr->PackHomInit());
         break;

         case eTMA_Ori :
         {
              if ((aTol<1e10) || (aFiltre !=0))
              {
                  GotOut = FiltragePointHomologues(pMatr->PackHomInit(),aNewPack,aTol,aFiltre);
                  aPackIn = &  aNewPack;
              }
              SauvHomologues(pMatr->PackHomInit());
	      if (mModele.AffineOrient().Val())
	      {
                 cMA_AffineOrient  aMAAO
                                (
                                    mAppli,
                                    *this,
                                    mGeoTer,
                                    mModele.L1CalcOri().Val(),
                                    pMatr->Sz(),
                                    *aPackIn
                                 );
              // SauvHomologues(*aPackIn);
                 if (mModele.MakeImagePxRef().Val())
                 {
                    aMAAO.MakeImagePxRef();
                 }
	      }
         }
         break;


	 case eTMA_Nuage3D :
	 {
	     std::string aNameRes =   std::string("Nuage3D")
	                            + mAppli.NameChantier() 
				    + std::string(".tif");
	     if (mModele.KeyNuage3D().IsInit())
	     {
	        aNameRes = mAppli.ICNM()->Assoc1To1
		           (
                               mModele.KeyNuage3D().Val(),
                               mAppli.NameChantier(),
                               true
                           );
	     }
	     aNameRes = mAppli.FullDirResult() + aNameRes;
	     Tiff_Im aFile
	             (
		         aNameRes.c_str(),
			 pMatr->ImAppX().sz(),
			 GenIm::real4,
			 Tiff_Im::No_Compr,
			 Tiff_Im::PtDAppuisDense
		     );
             ELISE_COPY
	     (
	          aFile.all_pts(),
		  Virgule
		  (
                      pMatr->ImPds().in(),
                      pMatr->ImAppX().in(),
                      pMatr->ImAppY().in(),
                      pMatr->ImAppZ().in()
		  ),
		  aFile.out()
	     );

	 }

         break;


         default :
            ELISE_ASSERT(false,"TypeModeleAnalytique Non Traite");
         break;
    }
    delete pMatr;
    if (GotOut)
    {
         mAppli.MicMacErreur
         (
            eErrPtHomHorsImage,
            "Point Homologue Hors Image",
            "Specification Utilisateur sur la Tolerance : <TolerancePointHomInImage>"
         );
    }
}

const  cOneModeleAnalytique  &  cModeleAnalytiqueComp::Modele() const 
{
   return mModele;
}


                 // ACCESSEURS


cElRegex_Ptr & cModeleAnalytiqueComp::AutomExport()
{
   return mAutomExport;
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
