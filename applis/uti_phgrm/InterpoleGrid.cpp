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
#include "private/all.h"
#include "XML_GEN/all.h"
#include "im_tpl/image.h"

using namespace NS_ParamChantierPhotogram;
using namespace NS_SuperposeImage;

/*
  Extrait un modele radial d'une image de points homologues
*/

namespace NS_InterpoleGrid
{

cDbleGrid::cXMLMode  aXMLM;

class cGridComp
{
    public :
       cGridComp
       (
           const std::string & aDir,
           const std::string & aName,
	   double aPds
       )  :
          mGrid    (aXMLM,aDir,aName),
	  mPds     (aPds),
	  mF       (mGrid.Focale()),
	  mPP      (mGrid.PP()),
	  mBox     (mGrid.P0_Dir(),mGrid.P1_Dir())
       {
       }

       double F()  {return mGrid.Focale();}
       Pt2dr  PP() {return mGrid.PP();}

       void AddEquation(double aFInit,const Pt2dr & aPPInit,
                        cEqDirecteDistorsion &,const Pt2dr & aPIm);

       const Box2dr & Box() {return mBox;}
       Pt2dr Step() {return mGrid.Step_Dir();}
    private  :

       cDbleGrid     mGrid;
       double        mPds;
       double        mF;
       Pt2dr         mPP;
       Box2dr        mBox;
};

void cGridComp::AddEquation
     (
           double aFInit,
	   const Pt2dr & aPPInit,
	   cEqDirecteDistorsion & anEDD,
	   const Pt2dr & aPIm0
     )
{
    Pt2dr aPIm =  aPPInit + (aPIm0-aPPInit) * (mF/aFInit);

    if (! mBox.inside(aPIm))
       return;

    Pt2dr aPPhgr = mGrid.Direct(aPIm);

    anEDD.AddObservation(aPIm0,aPPhgr,mPds);
}


class cInterpoleGridCal
{
    public :

        cInterpoleGridCal(const cInterpoleGrille &);
	void AllItere();
	void Sauv();

    private  :

        void OneItere();

        
        cInterpoleGrille mParam;
        double           mF0;
        double           mF1;
        double           mF2;

	double           mPds1;
	double           mPds2;
	double           mPdsTot;
      

        cGridComp       mGr1;
        cGridComp       mGr2;
	Box2dr          mBox;

        double           mFocaleInit;
	Pt2dr            mPPCInit;  // PP et Centre Dist sont confondu
        ElDistRadiale_PolynImpair mDist;
        cSetEqFormelles  mSetEq;
        cParamIFDistRadiale * mPIF;
        cEqDirecteDistorsion * mEqD;

};


cInterpoleGridCal::cInterpoleGridCal(const cInterpoleGrille & anIG) :
   mParam       (anIG),
   mF0          (anIG.Focale0()),
   mF1          (anIG.Focale1()),
   mF2          (anIG.Focale2()),

   mPds1        ((mF2-mF0) / (mF2-mF1)),
   mPds2        (1-mPds1),
   mPdsTot      (mPds1+mPds2),
   mGr1         (anIG.Directory().Val(),anIG.Grille1(),mPds1),
   mGr2         (anIG.Directory().Val(),anIG.Grille2(),mPds2),
   mBox         (mGr1.Box()),
   mFocaleInit  ((mGr1.F()*mPds1+mGr2.F()*mPds2)/mPdsTot),
   mPPCInit     ((mGr1.PP()*mPds1+mGr2.PP()*mPds2)/mPdsTot),
   mDist        (ElDistRadiale_PolynImpair::DistId(1.3*euclid(mPPCInit),mPPCInit,5)),
   mSetEq       (cNameSpaceEqF::eSysPlein,1000),
   mPIF         (mSetEq.NewIntrDistRad (mFocaleInit,mPPCInit,0,mDist)),
   mEqD         (mSetEq.NewEqDirecteDistorsion(*mPIF,cNameSpaceEqF::eTEDD_Interp))

{
    mPIF->SetFocFree(true);
    mPIF->SetLibertePPAndCDist(false,false);
    mSetEq.SetClosed();

    Box2dr aB2 = mGr2.Box();

    ELISE_ASSERT
    (
        (mBox._p0==aB2._p0)&&(mBox._p1==aB2._p1),
	"Diff box in cInterpoleGridCal"
    );
}

void cInterpoleGridCal::OneItere()
{
   mSetEq.AddContrainte(mPIF->StdContraintes());

   int aNb = mParam.NbPtsByIter().Val();

   Pt2di aPI;
   for (aPI.x=0; aPI.x<= aNb ; aPI.x++)
   {
       for (aPI.y=0; aPI.y<= aNb ; aPI.y++)
       {
            Pt2dr aP = mBox.FromCoordBar(Pt2dr(aPI)/double(aNb));
	    mGr1.AddEquation(mFocaleInit,mPPCInit,*mEqD,aP);
	    mGr2.AddEquation(mFocaleInit,mPPCInit,*mEqD,aP);
       }
   }
   mSetEq.SolveResetUpdate();

}

void cInterpoleGridCal::AllItere()
{
  for (int aK=0 ; aK<3 ; aK++)
        OneItere();

   std::cout << "END Focale \n\n";

   for (int aD = 1 ; aD <= mParam.DegPoly().Val() ; aD++)
   {
          mPIF->SetDRFDegreFige(aD);
          for (int aK=0 ; aK<3 ; aK++)
                   OneItere();
          std::cout << "END Degre " << aD <<  " \n\n";
   }

   if (mParam.LiberteCPP().Val() >= eCPPLies)
   {
        mPIF->SetCDistPPLie();
	std::cout << "Centre = " << mPIF->CurPP() << "\n";
	for (int aK=0 ; aK<4 ; aK++)
        {
            OneItere();
        }
   }
   if (mParam.LiberteCPP().Val() >= eCPPLibres)
   {
        mPIF->SetLibertePPAndCDist(true,true);
   }
}

void cInterpoleGridCal::Sauv()
{
  ElDistortion22_Gen * aDistAnaly = mEqD->Dist(Pt2dr(0,0));

  Pt2dr aStep1 =  mParam.StepGrid().ValWithDef(mGr1.Step());
  aDistAnaly->SaveAsGrid
  (
        mParam.Directory().Val()+mParam.Grille0(),
        mBox._p0,
        mBox._p1,
        aStep1
  );

}


};

using namespace NS_InterpoleGrid;



int main(int argc,char ** argv)
{
    ELISE_ASSERT(argc>=2,"Not Enough arg");
    cInterpoleGrille aParam = StdGetObjFromFile<cInterpoleGrille>
                            (
				   argv[1],
			           "include/XML_GEN/ParamChantierPhotogram.xml",
				   "InterpoleGrille",
				   "InterpoleGrille"
			    );

   cInterpoleGridCal anIGC(aParam);
   anIGC.AllItere();
   anIGC.Sauv();

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
