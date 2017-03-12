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
//#include "TpPPMD.h"


/********************************************************************/
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/

/*
   1- Hello word
   2- Lire string points d'appuis + points de liaison
   3- Creer une classe cAppli_BasculeRobuste
   4- Decouvrir le XML, les classe C++, "SetOfMesureAppuisFlottants"

        * include/XML_GEN/ParamChantierPhotogram.xml
        * include/XML_GEN/ParamChantierPhotogram.h

   5-  Ajout orientation + pattern

   6-  Creer les points 3D
*/

class cImageBasculeRobuste;
class cPointBascRobust;
class cAppli_BasculeRobuste;

class cImageBasculeRobuste
{
    public :
        cImageBasculeRobuste
        (
             const std::string  & aNameIm,
             cBasicGeomCap3D *    aCam
        ) :
          mNameIm (aNameIm),
          mCam    (aCam)
        {
        }

        std::string         mNameIm;
        cBasicGeomCap3D *   mCam;
};

class cPointBascRobust
{
    public :
        cPointBascRobust(const std::string & aNamePt,const Pt3dr & aPTer) :
           mNamePt (aNamePt),
           mPTer   (aPTer)
        {
        }

        std::string                            mNamePt;
        Pt3dr                                  mPTer;
        Pt3dr                                  mPInter;
        std::vector<cImageBasculeRobuste *>    mIms;
        std::vector<Pt2dr >                    mPtIms;
        std::vector<ElSeg3D >                  mSegs;
        
};



class cAppli_BasculeRobuste
{
    public :
       cAppli_BasculeRobuste(int argc,char ** argv);
    private :

       std::string                   mPatIm;
       std::string                   mOri;

       cSetOfMesureAppuisFlottants   mXML_MesureIm;
       cDicoAppuisFlottant           mXML_MesureTer;
       cElemAppliSetFile             mSetFile;
       cInterfChantierNameManipulateur *            mICNM;

       std::map<std::string,cImageBasculeRobuste *> mDicoIm;
       std::map<std::string,cPointBascRobust *>     mDicoPt;
       std::vector<cPointBascRobust *>              mVecPt;

       int                                          mNbPts;
       int                                          mNbTirage;
};

cAppli_BasculeRobuste::cAppli_BasculeRobuste(int argc,char ** argv) :
   mNbTirage  (1000)
{
	std::string aName2D,aName3D;
	
	ElInitArgMain
	(
	     argc,argv,
	     LArgMain() 
	                << EAMC(mPatIm,"Pattern des images")
	                << EAMC(mOri,"Orientation ")
	                << EAMC(aName3D,"Nom du fichier des points terrain")
	                << EAMC(aName2D,"Nom du fichier des mesure im"),
	     LArgMain()  << EAM(mNbTirage,"NbTir",true,"Nombre de tirage du Ransac")    
	);
	
	mSetFile.Init(mPatIm);
	mICNM = mSetFile.mICNM;
	const std::vector<std::string> * aVName = mSetFile.SetIm();
	std::cout << "Nbre Image " << aVName->size() << "\n";
	
    StdCorrecNameOrient(mOri,mSetFile.mDir);
	          
	       
     mXML_MesureTer = StdGetFromPCP(aName3D,DicoAppuisFlottant);
     mXML_MesureIm =  StdGetFromPCP(aName2D,SetOfMesureAppuisFlottants);
                
     for
     (
        std::list<cOneAppuisDAF>::const_iterator itAp=mXML_MesureTer.OneAppuisDAF().begin();
        itAp!=mXML_MesureTer.OneAppuisDAF().end();
        itAp++
     )
     {
		 cPointBascRobust * aNewP = new cPointBascRobust(itAp->NamePt(),itAp->Pt());
		 mDicoPt[itAp->NamePt()] = aNewP;
		 mVecPt.push_back(aNewP);
		// std::cout << "Name " << itAp->NamePt() << "\n";
	 }
	 
	 for (int aKN=0 ; aKN<int(aVName->size()) ; aKN++)
	 {
		 const std::string & aName = (*aVName)[aKN];
		 cBasicGeomCap3D * aCamGen = mICNM->StdCamGenerikOfNames(mOri,aName);
		 cImageBasculeRobuste * aCBR = new cImageBasculeRobuste(aName,aCamGen);
		 mDicoIm[aName] = aCBR;
	 }
    
     for 
     (
       std::list<cMesureAppuiFlottant1Im>::const_iterator it1M=mXML_MesureIm.MesureAppuiFlottant1Im().begin();
       it1M!=mXML_MesureIm.MesureAppuiFlottant1Im().end();
       it1M++
     )
     {
		 cImageBasculeRobuste * aCBR = mDicoIm[it1M->NameIm()];
		 ELISE_ASSERT(aCBR!=0,"Image absente du dictionnaire");
		 
		 for 
		 (
		     std::list<cOneMesureAF1I>::const_iterator itP=it1M->OneMesureAF1I().begin();
		     itP!=it1M->OneMesureAF1I().end();
		     itP++
		 ) 
		 {
			 cPointBascRobust * aPt = mDicoPt[itP->NamePt()];
			 ELISE_ASSERT(aPt!=0,"Point absent du dictionnaire"); 
			 aPt->mSegs.push_back(aCBR->mCam->Capteur2RayTer(itP->PtIm()));
			 aPt->mIms.push_back(aCBR);
			 aPt->mPtIms.push_back(itP->PtIm());
		 }
	 }
	 
	 for (int aKP=0 ; aKP<int(mVecPt.size()) ; aKP++)
	 {
		 cPointBascRobust * aPt = mVecPt[aKP];
		 
		 if (aPt->mSegs.size() >=2)
		 {
			 bool Ok;
			 aPt->mPInter = InterSeg(aPt->mSegs,Ok);
			 if (Ok)
			 {
				 std::cout << "For pt= " << aPt->mNamePt << "\n";
				 for (int aKIm=0 ; aKIm <int(aPt->mPtIms.size()) ; aKIm++)
				 {
					 Pt2dr aPInit = aPt->mPtIms[aKIm];
					 Pt2dr aPproj = aPt->mIms[aKIm]->mCam->Ter2Capteur(aPt->mPInter);
					 std::cout << "   Dist=" << euclid(aPInit,aPproj) 
					           << " for " << aPt->mIms[aKIm]->mNameIm << "\n";
				 }
			 }
		 }
	 }
	 
	 MakeFileXML(mXML_MesureTer,"TestReecriture.xml");
	 
}

int BasculeRobuste_main(int argc,char ** argv)
{
	std::cout << "Bienvenue a MicMac programmeur\n";
	cAppli_BasculeRobuste anAppli(argc,argv);
	return EXIT_SUCCESS;
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
