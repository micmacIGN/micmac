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

     /*****************************************/
     /*                                       */
     /*      cSurfAnalIdent                   */
     /*                                       */
     /*****************************************/

class cSurfAnalReperCart : public cInterfSurfaceAnalytique
{
    public :
        cSurfAnalReperCart (const cChCoCart & aCart) :
                 cInterfSurfaceAnalytique (true) ,
                 mCCCE2L                  (aCart.Inv()),
                 mCCCL2E                  (mCCCE2L.Inv())
         {
         }

        Pt3dr E2UVL(const Pt3dr & aP) const {return mCCCE2L.FromLoc(aP);}
        Pt3dr UVL2E(const Pt3dr & aP) const {return mCCCL2E.FromLoc(aP);}
        void AdaptBox(Pt2dr & aP0,Pt2dr & aP1) const {}

        cXmlDescriptionAnalytique Xml()  const
        {
             ELISE_ASSERT(false,"cSurfAnalIdent::Xml");
             cXmlDescriptionAnalytique aNS;
             return aNS;
        }

        bool HasOrthoLoc() const {return false;}

        std::vector<cInterSurfSegDroite>  InterDroite(const ElSeg3D & aSegOri,double aZ1) const 
        {
            ElSeg3D aSeg(E2UVL(aSegOri.PtOfAbsc(0)),E2UVL(aSegOri.PtOfAbsc(1)));
            std::vector<cInterSurfSegDroite> aRes;

            double aZ0 = aSeg.P0().z ;
            double aDZ = aSeg.TgNormee().z;

            if (aDZ==0) return aRes;

            aRes.push_back
            (
                cInterSurfSegDroite
                (
                    (aZ1-aZ0)/aDZ,
                    (  aZ0 >  aZ1 ) ? eSurfVI_Rent : eSurfVI_Sort
                )
            );
            return aRes;
        }
    private :

         cChCoCart mCCCE2L;
         cChCoCart mCCCL2E;

};


cInterfSurfaceAnalytique * cInterfSurfaceAnalytique::FromCCC(const cChCoCart & aCCC)
{
    cInterfSurfaceAnalytique * aRes = new cSurfAnalReperCart(aCCC);
    return aRes;
}

     /*****************************************/
     /*                                       */
     /*      cSurfAnalIdent                   */
     /*                                       */
     /*****************************************/

class cSurfAnalIdent : public cInterfSurfaceAnalytique
{
    public :
        cSurfAnalIdent(double aZRef) : 
                 cInterfSurfaceAnalytique (true) ,
                 mZRef                    (aZRef),
                 mVec                     (0,0,mZRef)
         {
         }

        Pt3dr E2UVL(const Pt3dr & aP) const {return aP - mVec;}
        Pt3dr UVL2E(const Pt3dr & aP) const {return aP + mVec;}
        void AdaptBox(Pt2dr & aP0,Pt2dr & aP1) const {}

        cXmlDescriptionAnalytique Xml()  const
        {
             ELISE_ASSERT(false,"cSurfAnalIdent::Xml");
             cXmlDescriptionAnalytique aNS;
             return aNS;
        }

        bool HasOrthoLoc() const {return false;}

        std::vector<cInterSurfSegDroite>  InterDroite(const ElSeg3D & aSeg,double aZ1) const 
        {
            std::vector<cInterSurfSegDroite> aRes;

            double aZ0 = aSeg.P0().z -mZRef;
            double aDZ = aSeg.TgNormee().z;

            if (aDZ==0) return aRes;

            aRes.push_back
            (
                cInterSurfSegDroite
                (
                    (aZ1-aZ0)/aDZ,
                    (  aZ0 >  aZ1 ) ? eSurfVI_Rent : eSurfVI_Sort
                )
            );
            return aRes;
        }
    private :
       double mZRef;
       Pt3dr  mVec;

};

cInterfSurfaceAnalytique * cInterfSurfaceAnalytique::Identite(double aZRef)
{
    static cInterfSurfaceAnalytique * aRes = new cSurfAnalIdent(aZRef);
    return aRes;
}


     /*****************************************/
     /*                                       */
     /*      cProjOrthoCylindrique            */
     /*                                       */
     /*****************************************/


cProjOrthoCylindrique::cProjOrthoCylindrique
(
    const cChCoCart & aL2A,
    const ElSeg3D & aSegAbs,
    bool    aAngulCorr
)  :
   cInterfSurfaceAnalytique (true),
   mL2A     (aL2A),
   mA2L     (mL2A.Inv()),
   mSegAbs     (aSegAbs),
   mAngulCorr  (aAngulCorr)
{
    ElSeg3D aSegLoc
            (
                 Ab2Loc(aSegAbs.PtOfAbsc(0)),
                 Ab2Loc(aSegAbs.PtOfAbsc(1))
            );

    mDist = ElAbs(aSegLoc.P0().z);
    
    Pt3dr aTg = aSegLoc.TgNormee();
    if (aTg.x <0)
       aTg = - aTg;

   mB = aTg.y / aTg.x;
   mC = aTg.z / aTg.x;

}

Pt3dr cProjOrthoCylindrique::Loc2Abs(const Pt3dr & aP) const
{
    return mL2A.FromLoc(aP);
}


Pt3dr cProjOrthoCylindrique::Ab2Loc(const Pt3dr & aP) const
{
   return mA2L.FromLoc(aP);
}

Pt3dr cProjOrthoCylindrique::Loc2Cyl(const Pt3dr  & aP) const
{
   if (mUnUseAnamXCSte) return aP;
// std::cout <<  "L2 C" << aP.y <<  " " << (aP.z-mD) << " " << (aP.y/(aP.z-mD))  << " " <<  atan2(aP.y,aP.z-mD) << "\n";

   double  A = aP.y -mB * aP.x;
   double  B = mDist + mC * aP.x - aP.z;


   double aV =  (mAngulCorr ?  atan2(A,B): (A/B)) ;

   return Pt3dr(aP.x,mDist * aV,aP.z);
}

Pt3dr cProjOrthoCylindrique::Cyl2Loc(const Pt3dr  & aP) const
{
   if (mUnUseAnamXCSte) return aP;

   double aV = aP.y / mDist;
   if (mAngulCorr) 
      aV = tan(aV);
   return Pt3dr
          (
             aP.x,
             mB * aP.x +  aV*(mDist +mC * aP.x -aP.z),
             aP.z
          );
   // return Pt3dr(aP.x,aP.y*(aP.z-mD)/mD,aP.z);
}

Pt3dr cProjOrthoCylindrique::E2UVL(const Pt3dr & aP) const
{
   return Loc2Cyl(Ab2Loc(aP));
}

Pt3dr cProjOrthoCylindrique::UVL2E(const Pt3dr & aP) const
{
   return Loc2Abs(Cyl2Loc(aP));
}


bool cProjOrthoCylindrique::HasOrthoLoc() const 
{
   return true;
}
 

Pt3dr cProjOrthoCylindrique::ToOrLoc(const Pt3dr & aP) const 
{
   return Cyl2Loc(aP);
}

Pt3dr cProjOrthoCylindrique::FromOrLoc(const Pt3dr & aP) const 
{
   return Loc2Cyl(aP);
}

bool  cProjOrthoCylindrique::OrthoLocIsXCste() const 
{
    return true;
}

bool cProjOrthoCylindrique::IsAnamXCsteOfCart() const { return true; }



//         Pt3dr FromOrLoc(const Pt3dr & aP) const ; // Def Err fatale


std::vector<cInterSurfSegDroite>  cProjOrthoCylindrique::InterDroite(const ElSeg3D & aSeg0,double aZ1) const 
{
    ElSeg3D aSeg(Ab2Loc(aSeg0.PtOfAbsc(0)),Ab2Loc(aSeg0.PtOfAbsc(1)));
    std::vector<cInterSurfSegDroite> aRes;

    double aZ0 = aSeg.P0().z;
    double aDZ = aSeg.TgNormee().z;

    if (aDZ==0) return aRes;


    aRes.push_back
    (
        cInterSurfSegDroite
        (
            (aZ1-aZ0)/aDZ,
            (  aZ0 >  aZ1 ) ? eSurfVI_Rent : eSurfVI_Sort
        )
    );
    return aRes;
}

void cProjOrthoCylindrique::AdaptBox(Pt2dr & aP0,Pt2dr & aP1) const
{
}

cXmlOrthoCyl cProjOrthoCylindrique::XmlOCyl() const
{
   cXmlOrthoCyl aRes;
   aRes.Repere() =  mL2A.El2Xml();
   aRes.P0() = mSegAbs.P0();
   aRes.P1() = mSegAbs.P1();
   aRes.AngulCorr() = mAngulCorr;

   return aRes;
}


cXmlDescriptionAnalytique cProjOrthoCylindrique::Xml() const
{
    cXmlDescriptionAnalytique aRes;
    aRes.OrthoCyl().SetVal(XmlOCyl());
    return aRes;
}

#define  NS_SuperposeImage 

cProjOrthoCylindrique cProjOrthoCylindrique::FromXml
                      (
                             const cXmlOneSurfaceAnalytique&,
                             const NS_SuperposeImage::cXmlOrthoCyl&  anOC
                      )
{
    cProjOrthoCylindrique aRes
           (
                 cChCoCart::Xml2El(anOC.Repere()),
                 ElSeg3D(anOC.P0(),anOC.P1()),
                 anOC.AngulCorr()
           );

/*
    cProjOrthoCylindrique aCAng
           (
                 anOC.Repere().Ori(),
                 anOC.Repere().Ox(),
                 anOC.Repere().Oy(),
                 anOC.Repere().Oz(),
                 anOC.Dist(),
                 true
           );
    cProjOrthoCylindrique aCOr
           (
                 anOC.Repere().Ori(),
                 anOC.Repere().Ox(),
                 anOC.Repere().Oy(),
                 anOC.Repere().Oz(),
                 anOC.Dist(),
                 false
           );


     while (1)
     {
          std::cout << "AAAAA \n";
          Pt3dr aP;
          std::cin >> aP.x >> aP.y >> aP.z;

          Pt3dr aQA = aCAng.UVL2E(aP);
          Pt3dr aRA = aCAng.E2UVL(aQA);
          Pt3dr aQO = aCOr.UVL2E(aP);
          Pt3dr aRO = aCOr.E2UVL(aQO);

          std::cout << aP << aRA << "  ;; " << aQA << "\n";
          std::cout << aP << aRO << "  ;; " << aQO << "\n";
     }
*/

     return aRes;
     
}



/*
*/



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
