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

bool ShowTore = false;


     /*****************************************/
     /*                                       */
     /*      cProjTore                        */
     /*                                       */
     /*****************************************/


cProjTore::cProjTore(const cCylindreRevolution & aCyl,const Pt3dr & aPEuclInitDiamTor) :
    cInterfSurfaceAnalytique(aCyl.E2UVL(aPEuclInitDiamTor).z>0),
    mCyl (aCyl),
    mAngulCorr (true)
{


    Pt3dr aDiamCyl =  mCyl.E2UVL(aPEuclInitDiamTor);
    aDiamCyl.x = 0;                     // On met le teta a 0 : alligne sur POnCyl
    mDiamEucl =  mCyl.UVL2E(aDiamCyl);
    Pt3dr aCylPOnCyl(0,aDiamCyl.y,0);
    Pt3dr anEuclPOnCyl  = mCyl.UVL2E(aCylPOnCyl);

    mCyl = cCylindreRevolution(IsVueExt(),mCyl.Axe(),anEuclPOnCyl);

    mDiamCyl = mCyl.E2UVL(mDiamEucl);

    if (ShowTore) 
    {
       std::cout << " DiamEuclInit " << aPEuclInitDiamTor << " " << mDiamEucl << "\n";
       std::cout << " P0 " << mCyl.P0() << "\n";
    }
}

//  =========================  COORDONNEE =========================
/*
     Axe       Cyl
      |        ||
      |        ||    Y
      |        ||    |
      |        ||    |
      |        ||    |
      |        ||  -------- * Diam
                     Z
                     
*/

Pt3dr cProjTore::Cyl2Tore(const Pt3dr & aP) const // Loc2Cyl
{
   if (mUnUseAnamXCSte) return aP;

   double  A = aP.y ;  // le Z avec origine sur le diam du tore
   double  B = mDiamCyl.z-aP.z  ;


   double aV =  (mAngulCorr ?  atan2(A,B): (A/B)) ;

   return Pt3dr(aP.x,mDiamCyl.z * aV,aP.z);

}
Pt3dr cProjTore::FromOrLoc(const Pt3dr & aP) const { return Cyl2Tore(aP); }
Pt3dr cProjTore::E2UVL(const Pt3dr & aP) const { return Cyl2Tore(mCyl.E2UVL(aP)); }



Pt3dr cProjTore::Tore2Cyl(const Pt3dr  & aP) const  // :Cyl2Loc : ToOrLoc
{
   if (mUnUseAnamXCSte) return aP;
   double aV = aP.y / mDiamCyl.z;
   if (mAngulCorr)
      aV = tan(aV);
   // A/B = V  ; aP.y = A = BV = V(mDiamCyl.z-aP.z)
   return Pt3dr
          (
             aP.x,
             aV * (mDiamCyl.z-aP.z),
             aP.z
          );
   // return Pt3dr(aP.x,aP.y*(aP.z-mD)/mD,aP.z);
}
Pt3dr cProjTore::ToOrLoc(const Pt3dr & aP) const { return Tore2Cyl(aP); }
Pt3dr cProjTore::UVL2E(const Pt3dr & aP) const { return mCyl.UVL2E(Tore2Cyl(aP)); }

std::vector<cInterSurfSegDroite>  cProjTore::InterDroite(const ElSeg3D & aSeg,double aZ0) const 
{
   return mCyl.InterDroite(aSeg,aZ0);
}
  



//  =========================  DIVERS =========================

bool  cProjTore::HasOrthoLoc()     const {return true;}
bool  cProjTore::OrthoLocIsXCste() const {return true;}

void cProjTore::AdaptBox(Pt2dr & aP0,Pt2dr & aP1) const 
{
   mCyl.AdaptBox(aP0,aP1);
}


//  =========================  XML =========================

cXmlToreRevol   cProjTore::XmlTore() const
{
    cXmlToreRevol aRes;

    aRes.Cyl() = mCyl.XmlCyl();
    aRes.POriTore() =  mDiamEucl;

    return aRes;
}
cXmlDescriptionAnalytique cProjTore::Xml() const
{
    cXmlDescriptionAnalytique aRes;
    aRes.Tore().SetVal(XmlTore());
    return aRes;
}

// cProjTore(const cCylindreRevolution & aCyl,const Pt3dr & aPEuclDiamTor);

cProjTore  cProjTore::FromXml(const cXmlOneSurfaceAnalytique & aXmlSA,const cXmlToreRevol & aXmlTore)
{
    return cProjTore
           (
                cCylindreRevolution::FromXml(aXmlSA,aXmlTore.Cyl()),
                aXmlTore.POriTore()
           );
}

cXmlModeleSurfaceComplexe  cProjTore::SimpleXml(const std::string & anIdAux) const
{
    cXmlModeleSurfaceComplexe aRes = cInterfSurfaceAnalytique::SimpleXml("TheSurf");
    cXmlModeleSurfaceComplexe aRCyl = mCyl.SimpleXml("TheSurfAux");

    aRes.XmlOneSurfaceAnalytique().push_back(*(aRCyl.XmlOneSurfaceAnalytique().begin()));

    return aRes;
}

//  =================== Creation d'un tore =============================

class cAppliDonuts : cAppliWithSetImage
{
     public :
          cAppliDonuts(int argc,char **argv);
          void NoOp() {}

     private :
          std::string   mFullName;
          std::string   mOri;
          std::string   mNameCyl;
          std::string   mOut;
          bool          mShow;
          bool          mCheck;
};


cAppliDonuts::cAppliDonuts(int argc,char **argv) :
     cAppliWithSetImage(argc-1,argv+1,0),
     mShow  (false),
     mCheck (false)
{
     ElInitArgMain
     (
           argc,argv,
           LArgMain() << EAMC(mFullName,"Full name (Dir+Pat)", eSAM_IsPatFile)
                      << EAMC(mOri,"Orientation Dir")
                      << EAMC(mNameCyl,"Name of XML cylinder"),
           LArgMain() << EAM(mOut,"Out",true,"Out Put, Def = Tore_{NameCyl}")
                      << EAM(mShow,"Show",true,"Show details")
                      << EAM(mCheck,"Check",true,"Show details")
    );
    ShowTore = mShow;

    if (!EAMIsInit(&mOut))
      mOut = "Tore_"+ mNameCyl;


    cElXMLTree aTree(Dir()+mNameCyl);
    cElXMLTree * aTreeSurf = aTree.Get("XmlOneSurfaceAnalytique");
    cXmlOneSurfaceAnalytique aXmlSA;
    xml_init(aXmlSA,aTreeSurf);

    cXmlCylindreRevolution * aXmlCyl = aXmlSA.XmlDescriptionAnalytique().Cyl().PtrVal();
    ELISE_ASSERT(aXmlCyl!=0,"Cannot find XmlCylindreRevolution in Donnuts");
    
    cCylindreRevolution aCyl = cCylindreRevolution::FromXml(aXmlSA,*aXmlCyl);


    std::vector<double> aVTeta;
    std::vector<double> aVZ;
    std::vector<double> aVRho;
    for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
    {
         CamStenope * aCS = mVSoms[aK]->attr().mIma->mCam;
         Pt3dr aPE = aCS->PseudoOpticalCenter();
         Pt3dr aPCyl = aCyl.E2UVL(aPE);
         aVTeta.push_back(aPCyl.x);
         aVZ.push_back(aPCyl.y);
         aVRho.push_back(aPCyl.z);
         if (mShow)
            std::cout  << aPCyl << "Teta " << aPCyl.x / aCyl.Ray() << " Z " << aPCyl.y << " Rho " << aPCyl.z + aCyl.Ray() << "\n";
    }
    double aTetaMed = MedianeSup(aVTeta);
    double aZMed = MedianeSup(aVZ);
    double aRhoMed = MedianeSup(aVRho);

    Pt3dr aDiamCyl (aTetaMed,aZMed,aRhoMed);
    // std::cout << "P In Coord Cyl " << aPCyl << "\n";
    Pt3dr  aDiamEucl = aCyl.UVL2E(aDiamCyl);

    cProjTore aToreFin(aCyl,aDiamEucl);
    MakeFileXML(aToreFin.SimpleXml("TheSurfAux"),mOut);

    double IndQual = 0;
    int   aNbNonInter =0;
    if (mCheck)
    {
        cXmlModeleSurfaceComplexe aModele = StdGetFromSI(mOut,XmlModeleSurfaceComplexe);
        // cInterfSurfaceAnalytique & aTore =  aToreFin;
        cXmlOneSurfaceAnalytique aXSA  = *(aModele.XmlOneSurfaceAnalytique().begin());

        cInterfSurfaceAnalytique & aISA =  *(cInterfSurfaceAnalytique::FromXml(aXSA));
        for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
        {
             CamStenope * aCS = mVSoms[aK]->attr().mIma->mCam;
             Pt3dr aPE = aCS->ImEtProf2Terrain(Pt2dr(aCS->Sz())/2.0,aRhoMed);
             Pt3dr aPC = aISA.E2UVL(aPE);
             Pt3dr aPE2 = aISA.UVL2E(aPC);

             double aEps= 1e-5;
             Pt3dr aP0 (aPC.x,0,0);  // Le syste est Orthonorme sur le cyl au niveau du diam
             Pt3dr aGradT = (aISA.UVL2E(aP0+Pt3dr(aEps,0,0))-aISA.UVL2E(aP0+Pt3dr(-aEps,0,0)))/(2*aEps);
             Pt3dr aGradZ = (aISA.UVL2E(aP0+Pt3dr(0,aEps,0))-aISA.UVL2E(aP0+Pt3dr(0,-aEps,0)))/(2*aEps);
             Pt3dr aGradR = (aISA.UVL2E(aP0+Pt3dr(0,0,aEps))-aISA.UVL2E(aP0+Pt3dr(0,0,-aEps))) /(2*aEps);

             ElMatrix<double> aP = MatFromCol(aGradT,aGradZ,aGradR);
             aP = aP * aP.transpose() - ElMatrix<double>(3,true);
             double aMixte = scal(aGradT,aGradZ^aGradR);


             double EcartInv = euclid(aPE-aPE2);  // Test que les systeme sont inverse
             double EcartON = aP.L2();            // Test que le system est orthorme en R=0
             double EcartDir =  ElAbs(aMixte-1);  // Test que le systeme est direct


/*
             std::cout << " Tor[C] " << aISA.E2UVL(aCS->PseudoOpticalCenter()) 
                       << " Cyl[C] " << aCyl.E2UVL(aCS->PseudoOpticalCenter())
                       << "\n";
*/

             std::cout 
                       << "E2U o U2E: " << EcartInv  
                       << " Rot;" << EcartON        
                        << " Dir " << EcartDir     
                       ;

             ElSeg3D aSeg = aCS->Capteur2RayTer(aCS->Sz()/2.0);
             cTplValGesInit<Pt3dr> aTplPTer = aISA.InterDemiDroiteVisible(aSeg,0);
             if (aTplPTer.IsInit())
             {
                   Pt3dr aPTerE = aTplPTer.Val();
                   std::cout << " GGGG " << aPTerE;
             }
             else
             {

                std::cout  << " Int XXXXXX ";
                aNbNonInter++;
             }
             std::cout 
                       << "\n";

             IndQual = ElMax(IndQual,EcartInv + EcartON + EcartDir);
        }
        std::cout << "=============================================\n";
        std::cout << "   Qual = " << IndQual  << "\n";
        std::cout << "   NonInter = " << aNbNonInter  << "\n";
    }

    
}

int Donuts_main(int argc,char **argv)
{
    cAppliDonuts anAppli(argc,argv);
    anAppli.NoOp();

    return 1;
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
