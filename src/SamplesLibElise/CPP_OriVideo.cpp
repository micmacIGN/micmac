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


/*typedef enum
{
	eBBA,      //bundle block adjustment  ie Tapas
	eSBBA,     //structure-less BBA 	  ie Martini
	eSBBAFus   //SBBA and BBA intertwined ie Martini et Tapas

}eTypeOriVid;
*/


class cAppliOriVideo
{
	public :
		cAppliOriVideo(int argc, char** argv);

			
	private :
	 
		int                        mSzW;
		int				           mNbWinAll;		
		int				           mNbWinVois;		
		std::list< std::string >   mWName;

		std::string        mInOri;
		std::string  	   mDir;
		std::string        mIms;
		std::string        mSH;
		std::string        mOut;

		cInterfChantierNameManipulateur 			* mICNM; 
	    const cInterfChantierNameManipulateur::tSet * mSetIm;	
		int 										  mNbIm;

		bool        			mModeHelp;	
		std::string 			mStrType;	
		eTypeOriVid 			mType;

		void        ReadType    (const std::string & aType);	
		std::string MakeFenName (const int aNum);
		void        CalculFen   ();


		void DoBBA();
		void DoSBBA();
		void DoSBBAFus();	
		 
		
};

std::string cAppliOriVideo::MakeFenName (const int aNum)
{
	return "Fen-" + ToString(aNum) + ".xml";
}

void cAppliOriVideo::CalculFen()
{

	//calculate the windows
	std::vector< std::list< std::string >> aWVTmp;

	std::cout << "fenetres sous traitement";
	for (int aW=0; aW<mNbWinAll; aW++)
	{
		cListOfName              aXml;
		std::list< std::string > aImInWL;

		for (int aIm=0; aIm<mSzW; aIm++)
		{	
			int aIGlob = aW*mSzW + aIm;
		
			if (aIGlob<mNbIm)
				aImInWL.push_back((*mSetIm)[aIGlob]);

		}
		std::cout << ".." << aW << "/" << mNbWinAll ;
		
		aWVTmp.push_back(aImInWL);

		aXml.Name() = aImInWL;
		MakeFileXML(aXml,MakeFenName(aW));
		mWName.push_back(MakeFenName(aW));
	}


	//add a "tail" to each window    --- move the xml save outside
	/*if (mNbWinVois)	
	{
		for (int aW=mNbWinAll-1; aW>=0; aW--)
		{
			aWinsV->at(aW) 	
		}
	}
	*/
	
	

	
}


void cAppliOriVideo::DoBBA()
{
	for (auto aW : mWName)
	{
		std::string aCom = MMBinFile("mm3d Tapas RadialStd ") 
                           + "NKS-Set-OfFile@" + aW 
                           + " InOri=" + mInOri + " Out=" + mOut;

		std::cout << "aCom=" << aCom << "\n";

		TopSystem(aCom.c_str());
	}	
}

void cAppliOriVideo::DoSBBA()
{}

void cAppliOriVideo::DoSBBAFus()
{}


cAppliOriVideo::cAppliOriVideo(int argc, char** argv) :
	mSzW(5),
	mNbWinVois(2),
	mSH("")
{
	std::string aPattern;

	ElInitArgMain
	(
		argc, argv,
		LArgMain() << EAMC(mStrType,"Orientation mode (enum values)")
		           << EAMC(aPattern,"Pattern of images")
				   << EAMC(mSzW,"Processing window size"),
		LArgMain() << EAM (mNbWinVois,"NbW",true,"No of windows estimated at ti; Def=2")  
				   << EAM (mInOri,"InOri",true,"Input orientation")  
				   << EAM (mSH,"SH",true,"Homol prefix") 
	);

	
#if (ELISE_windows)
      replace( aPattern.begin(), aPattern.end(), '\\', '/' );
#endif
    SplitDirAndFile(mDir,mIms,aPattern);
    StdCorrecNameOrient(mInOri,mDir);

	ReadType(mStrType);


	if (EAMIsInit(&mInOri))
		mOut = mInOri;
	else
		mOut = "DIV";



	mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    mSetIm = mICNM->Get(mIms);
    mNbIm = (int)mSetIm->size();


	mNbWinAll = std::ceil(double(mNbIm)/mSzW);

	CalculFen();

	//InitDefValFromType
	switch (mType)
	{
		case eBBA :
			DoBBA();
			break;

		case eSBBA :
			DoSBBA();
			break;

		case eSBBAFus :
			DoSBBAFus();
			break;

		case eUndefVal :
			break;
	}
}

void cAppliOriVideo::ReadType(const std::string & aType)
{
    mStrType = aType;
    StdReadEnum(mModeHelp,mType,mStrType,eUndefVal);
/*	eTypeMalt   xType;
    StdReadEnum(mModeHelp,xType,mStrType,eNbTypesMNE);*/
}


int OriVideo_main(int argc,char** argv)
{
	cAppliOriVideo anAppli(argc,argv);

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




