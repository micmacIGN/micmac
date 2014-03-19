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

// List  of classes
class cMCI_Appli;
// Contains the information to store each image : Radiometry & Geometry
class cMCI_Ima;

// classes declaration

class cMCI_Ima
{
	public:
	   cMCI_Ima(cMCI_Appli & anAppli,const std::string & aName);
	
	   Pt2dr ClikIn();
	   
	   void  DrawFaisceaucReproj(cMCI_Ima & aMas,const Pt2dr & aP);
	   
	private :
	   cMCI_Appli &   mAppli;
	   std::string     mName;
	   Tiff_Im         mTifIm;
	   Pt2di           mSz;
	   Im2D_U_INT1     mIm;
	   Video_Win *     mW;
	   std::string     mNameOri;
	   CamStenope *    mCam;
	   
};



class cMCI_Appli
{
	public :
	
    	cMCI_Appli(int argc, char** argv);
    	const std::string & Dir() const {return mDir;}
    	bool ShowArgs() const {return mShowArgs;}
    	std::string NameIm2NameOri(const std::string &) const;
		cInterfChantierNameManipulateur * ICNM() const {return mICNM;}

        Pt2dr ClikInMaster();
    	
    	void TestProj();
    	
    private :
        cMCI_Appli(const cMCI_Appli &); // To avoid unwanted copies 
        
        void DoShowArgs();
    
    
    	std::string mFullName;
		std::string mDir;
		std::string mPat;
		std::string mOri;
		std::string mNameMast;
		std::list<std::string> mLFile;
		cInterfChantierNameManipulateur * mICNM;
		std::vector<cMCI_Ima *>          mIms;
		cMCI_Ima *                       mMastIm;
		bool                              mShowArgs;
};

/********************************************************************/
/*                                                                  */
/*         cMCI_Ima                                                 */
/*                                                                  */
/****************StdCorrecNameOrient****************************************************/
//CamStenope * CamOrientGenFromFile(const std::string & aNameFile,cInterfChantierNameManipulateur * anICNM);

cMCI_Ima::cMCI_Ima(cMCI_Appli & anAppli,const std::string & aName) :
   mAppli  (anAppli),
   mName   (aName),
   mTifIm  (Tiff_Im::StdConvGen(mAppli.Dir() + mName,1,true)),
   mSz     (mTifIm.sz()),
   mIm     (mSz.x,mSz.y),
   mW      (0),
   mNameOri (mAppli.NameIm2NameOri(mName)),
   mCam      (CamOrientGenFromFile(mNameOri,mAppli.ICNM()))
{
   ELISE_COPY(mIm.all_pts(),mTifIm.in(),mIm.out());
    
   if (mAppli.ShowArgs())
   {
	   std::cout << mName << mSz << "\n";
	   mW = Video_Win::PtrWStd(Pt2di(1200,800));
	   mW->set_title(mName.c_str());
	   ELISE_COPY(mW->all_pts(),mTifIm.in(),mW->ogray());
	   //mW->clik_in();
	   
	   ELISE_COPY(mW->all_pts(),255-mIm.in(),mW->ogray());
	   //mW->clik_in();
	   std::cout << mNameOri 
	             << " F=" << mCam->Focale()
	             << " P=" << mCam->GetProfondeur()
	             << " A=" << mCam->GetAltiSol()
	             << "\n";
   }
  
}
    	
Pt2dr cMCI_Ima::ClikIn()
{
	return mW->clik_in()._pt;
}
 	
void  cMCI_Ima::DrawFaisceaucReproj(cMCI_Ima & aMas,const Pt2dr & aP)
{
	double aProfMoy =  aMas.mCam->GetProfondeur();
	double aCoef = 1.2;
	
	std::vector<Pt2dr> aVProj;
	for (double aMul = 0.2; aMul < 5; aMul *=aCoef)
	{
		 Pt3dr aP3d =  aMas.mCam->ImEtProf2Terrain(aP,aProfMoy*aMul);
		 Pt2dr aPIm = this->mCam->R3toF2(aP3d);
		 
		 aVProj.push_back(aPIm);
    }
    for (int aK=0 ; aK<(aVProj.size()-1) ; aK++)
        mW->draw_seg(aVProj[aK],aVProj[aK+1],mW->pdisc()(P8COL::red));
}

/********************************************************************/
/*                                                                  */
/*         cMCI_Appli                                               */
/*                                                                  */
/********************************************************************/

cMCI_Appli::cMCI_Appli(int argc, char** argv)
{
	// Reading parameter : check and  convert strings to low level objects
	mShowArgs=false;
	ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mFullName,"Full Name (Dir+Pat)")
                    << EAMC(mNameMast,"Name of Master Image")
                    << EAMC(mOri,"Used orientation"),
        LArgMain()  << EAM(mShowArgs,"Show",true,"Give details on args")
    );
    
    // Initialize name manipulator & files 
    SplitDirAndFile(mDir,mPat,mFullName);
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    mLFile = mICNM->StdGetListOfFile(mPat);
    
    StdCorrecNameOrient(mOri,mDir);
    
    if (mShowArgs) DoShowArgs();
  
    // Initialize all the images structure
    mMastIm = 0;
    for (
              std::list<std::string>::iterator itS=mLFile.begin();
              itS!=mLFile.end();
              itS++
              )
     {
           cMCI_Ima * aNewIm = new  cMCI_Ima(*this,*itS);
           mIms.push_back(aNewIm);
           if (*itS==mNameMast)
               mMastIm = aNewIm;
     }	
     
     // Ckeck the master is included in the pattern
     ELISE_ASSERT
     (
	    mMastIm!=0,
	    "Master image not found in pattern"
     );
    
     if (mShowArgs)
        TestProj();
}

void cMCI_Appli::TestProj()
{
	while (1)
	{
		Pt2dr aP = ClikInMaster();
		for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
		{
			mIms[aKIm]->DrawFaisceaucReproj(*mMastIm,aP);
		}
	}
}  


Pt2dr cMCI_Appli::ClikInMaster()
{
	return mMastIm->ClikIn();
}

     
std::string cMCI_Appli::NameIm2NameOri(const std::string & aNameIm) const
{
	return mICNM->Assoc1To1
	(
	    "NKS-Assoc-Im2Orient@-"+mOri+"@",
	    aNameIm,
	    true
	);
}
      
void cMCI_Appli::DoShowArgs()
{
     std::cout << "DIR=" << mDir << " Pat=" << mPat << " Orient=" << mOri<< "\n"; 
     std::cout << "Nb Files " << mLFile.size() << "\n";
     for (
              std::list<std::string>::iterator itS=mLFile.begin();
              itS!=mLFile.end();
              itS++
              )
      {
              std::cout << "    F=" << *itS << "\n";
      }	
}

/********************************************************************/
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/

int ExoMCI_main(int argc, char** argv)
{
   cMCI_Appli anAppli(argc,argv);
   
   return EXIT_SUCCESS;
}


int ExoMCI_2_main(int argc, char** argv)
{
   std::string aNameFile;
   double D=1.0;
   
   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameFile,"Name of GCP file"),
        LArgMain()  << EAM(D,"D",true,"Unused")
    );

    cDicoAppuisFlottant aDico=  StdGetFromPCP(aNameFile,DicoAppuisFlottant);


    std::cout << "Nb Pts " <<  aDico.OneAppuisDAF().size() << "\n";
    std::list<cOneAppuisDAF> & aLGCP =  aDico.OneAppuisDAF();
     
    for (
             std::list<cOneAppuisDAF>::iterator iT= aLGCP.begin();
             iT != aLGCP.end();
             iT++
    )
    {
		// iT->Pt() equiv a (*iTp).Pt()
		std::cout << iT->NamePt() << " " << iT->Pt() << "\n";
	}

   return EXIT_SUCCESS;
}

int ExoMCI_1_main(int argc, char** argv)
{
   int I,J;
   double D=1.0;
   
   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(I,"Left Operand")
                    << EAMC(J,"Right Operand"),
        LArgMain()  << EAM(D,"D",true,"divisor of I+J")
    );

    std::cout << "(I+J)/D = " <<  (I+J)/D << "\n";

   return EXIT_SUCCESS;
}


int ExoMCI_0_main(int argc, char** argv)
{

   for (int aK=0 ; aK<argc ; aK++)
      std::cout << " Argv[" << aK << "]=" << argv[aK] << "\n";

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
