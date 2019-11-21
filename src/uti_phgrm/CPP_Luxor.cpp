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

extern bool ERupnik_MM();

/*typedef enum
{
	eBBA,      //bundle block adjustment  ie Tapas
	eSBBA,     //structure-less BBA 	  ie Martini
	eSBBAFus   //SBBA and BBA intertwined ie Martini et Tapas

}eTypeOriVid;
*/


/* to do :

fusion
- faire coherent les sorties de BBA et SBBA pourque ; peut-etre faudrait re-ecrire un peu pour que InOri soit mis plus globalement?
- 


*/


void BanniereLuxor()
{
    std::cout <<  "\n";
    std::cout <<  " *********************************************\n";
    std::cout <<  " *     L-inear                               *\n";
    std::cout <<  " *     U-ltra                                *\n";
    std::cout <<  " *     X (Q)uick (S)liding                   *\n";
    std::cout <<  " *     OR-ientation                          *\n";
    std::cout <<  " *********************************************\n\n";

}



class cAppliLuxor
{
	public :
		cAppliLuxor(int argc, char** argv);

			
	private :
	 
		int                        mWSz;        //sliding window size
		int				           mNbWinAll;	//number of all windows
		int				           mWPas;		//sliding window step (how many frames added at a time)
		int				           mLBAPas;		//bba step
		std::list< std::string >   mWName;      //list of all frames

		std::string        mInOri;
		std::string        mInCal;
		std::string  	   mDir;
		std::string        mIms;
		std::string        mSH;
		std::string        mOut;
        std::string        mWinPrefix;
        std::string        mWinOriPostfix;

        bool DoBASCULE;
        bool DoLBA; 

        bool LBA_ACTIVE;                        //to indicate whether adjustement has been lunched

		cInterfChantierNameManipulateur 			* mICNM; 
	    const cInterfChantierNameManipulateur::tSet * mSetIm;	
		int 										  mNbIm;

		bool        			mModeHelp;	
		std::string 			mStrType;	
		eTypeOriVid 			mType;

		void        ReadType        (const std::string & aType);	
		std::string MakeFenName     (const int aNum);
		std::string MakeFenOriName  (const int aNum);
		std::string MakeCampOriName (const std::string &);
		void        CalculFen       ();
        std::string WinPrefix       () {return mWinPrefix;}
        std::string WinOriPostfix   () {return mWinOriPostfix;}


		void DoBBAGlob     ();
		void DoBBA         ();
		void DoBBA         (const std::string &,bool Init=false);
		void DoSBBA        ();
		void DoSBBA        (const std::string & aNameCur,
                            const std::string & aNamePrev="");
		void BasculeMorito (const std::string & aNameCur,
                            const std::string & aNamePrev);
		void LBACampari    (const std::list<std::string>& );	
		 
		
};

std::string cAppliLuxor::MakeFenName (const int aNum)
{
	return WinPrefix() + ToString(aNum) + ".xml";
}

std::string cAppliLuxor::MakeFenOriName (const int aNum)
{
	return WinPrefix() + ToString(aNum) + WinOriPostfix() + ".xml";
}

std::string cAppliLuxor::MakeCampOriName (const std::string & aName)
{
	return aName + WinOriPostfix() + ".xml";
}


void cAppliLuxor::CalculFen()
{
    std::string aKeyOri2ImGen = std::string("NKS-Assoc-Im2Orient@-") + "Fen-";

	//calculate the windows
	//std::vector< std::list< std::string >> aWVTmp;

	std::cout << "fenetres sous traitement: \n";
	for (int aW=0; aW<mNbWinAll; aW++)
	{

        std::string aKeyOri2Im = aKeyOri2ImGen + ToString(aW);

		cListOfName              aXml;
		cListOfName              aXmlOri;
		std::list< std::string > aImInWL;
		std::list< std::string > aImInWLOri;



		for (int aIm=0; aIm<mWSz; aIm++)
		{	
			int aIGlob ;   
			aIGlob = aW * mWPas + aIm;
		    
			if (aIGlob<mNbIm)
            {
				aImInWL.push_back((*mSetIm)[aIGlob]);

                aImInWLOri.push_back( mICNM->Assoc1To1(aKeyOri2Im,(*mSetIm)[aIGlob],true));

			}
			std::cout << " Num=" << aIGlob << " " << (*mSetIm)[aIGlob] << " \n" ;

		}
		std::cout << ".." << aW+1 << "/" << mNbWinAll << "\n" ;
		
		//aWVTmp.push_back(aImInWL);

		aXml.Name()    = aImInWL;
		aXmlOri.Name() = aImInWLOri;

		MakeFileXML(aXml,MakeFenName(aW));
		MakeFileXML(aXmlOri,MakeFenOriName(aW));

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

void cAppliLuxor::LBACampari(const std::list<std::string>& aStrList)
{
    /* Read all files and create a new merged NKS
       - mmvii EditSet could do it but must be compiled 
         which shouldn't be taken for granted
       Copy the calibration files too  */
 
    mInOri = "Camp-"; 

    std::string aKeyOri2ImGen = std::string("NKS-Assoc-Im2Orient@-");

    std::string             aCampOriIn = "Ori-Campari-tmp";
    ELISE_fp::MkDir(aCampOriIn);

    std::string             aXmlName = "Camp-";
    cListOfName             aXml;
    std::list<std::string>  aListName;
    for (auto itL : aStrList)
    {
       std::string aKeyOri2Im = aKeyOri2ImGen + StdPrefix(itL);

       mInOri   += StdPrefix(itL) + "_";
       aXmlName += StdPrefix(itL) + "_";

       cListOfName aLTmp = StdGetFromPCP(itL,ListOfName); 
       for (auto itLT : aLTmp.Name())
       {
           aListName.push_back(itLT);

           ELISE_fp::CpFile
           (
               mICNM->Assoc1To1(aKeyOri2Im,itLT,true),
               aCampOriIn
           );
       }
       //calibration file
       ELISE_fp::CpFile
       (
           DirOfFile(mICNM->Assoc1To1(aKeyOri2Im,"",true)) + "AutoCal*.xml",
           aCampOriIn
       );

    } 
    aXml.Name() = aListName;
    MakeFileXML(aXml,aXmlName+".xml");

  
    /* Create NKS with orientations (to feed in Morito / Martini later on)  */
    std::string aXmlOriName = MakeCampOriName(aXmlName);
    cListOfName aXmlOri;
    std::list<std::string>  aListOriName;

    std::string aKeyOri2Im = aKeyOri2ImGen + aXmlName;
    for (auto it : aListName) 
    {
        aListOriName.push_back(mICNM->Assoc1To1(aKeyOri2Im,it,true));
    }
    aXmlOri.Name() = aListOriName;
    MakeFileXML(aXmlOri,aXmlOriName); 


    std::string aCom = MMBinFile("mm3d Campari ") +
                       "NKS-Set-OfFile@" + aXmlName + ".xml " 
                       + aCampOriIn + " " 
                       + mInOri ;

	if (EAMIsInit(&mSH))
        aCom += " SH=" + mSH;

    std::cout << "LBA= " << aCom << "\n";

    TopSystem(aCom.c_str());
   
    ELISE_fp::PurgeDirRecursif(aCampOriIn);
    ELISE_fp::RmDir(aCampOriIn);
    
    LBA_ACTIVE = true; 
}

void cAppliLuxor::BasculeMorito (const std::string & aNameCur,
                                 const std::string & aNamePrev)
{
	/*std::string aCom = MMBinFile("mm3d Morito ") +
                        "Ori-" + StdPrefix(aNameCur) + "/Ori.*xml " +
                        "Ori-" + StdPrefix(aNamePrev) + "/Ori.*xml " +
                         StdPrefix(aNameCur) ; */
    std::cout << "aNamePrev/aNameCur=" << aNamePrev << " " << aNameCur << "\n";                     
	std::string aCom = MMBinFile("mm3d Morito ") +
                        "NKS-Set-OfFile@" + StdPrefix(aNamePrev) + "-Ori.xml " + 
                        "NKS-Set-OfFile@" + StdPrefix(aNameCur)  + "-Ori.xml " +
                         StdPrefix(aNameCur) ; 
   
    
    std::cout << "CMD=" << aCom << "\n";                     
	TopSystem(aCom.c_str());
     

}


void cAppliLuxor::DoBBA(const std::string & aName,bool Init)
{


	std::string aCom = MMBinFile("mm3d Tapas Figee ") 
                       + "NKS-Set-OfFile@" + aName 
                       + " RefineAll=0 Out=" + mOut;

	if (EAMIsInit(&mInOri) && Init)	
		aCom += " InOri=" + mInOri;
	else if (!Init)
	{ 
		mInOri = mOut;
		aCom += " InOri=" + mInOri;

	}

	if (EAMIsInit(&mInCal))	
		aCom += " InCal=" + mInCal;

	//io figee
	//aCom += " LibFoc=0 DegRadMax=0 LibPP=0" ;  

	if (EAMIsInit(&mSH))	
		aCom += " SH=" + mSH;


	std::cout << "aCom=" << aCom << "\n";

	TopSystem(aCom.c_str());
}

void cAppliLuxor::DoBBA()
{
	int i=0;

	for (auto aW : mWName)
	{
		if (i==0)
			DoBBA(aW,true);
		else
			DoBBA(aW);
		i++;
	}	
}

void cAppliLuxor::DoSBBA(const std::string & aName,const std::string & aNamePrev)
{
    std::cout << "er aName/aNamePrev=" << aName << " " << aNamePrev << "\n";

    std::string aInOri = aNamePrev;
   // (aNamePrev == "") ? aInOri = aNamePrev : aInOri = StdPrefix(aNamePrev); 

	std::string aCom   = MMBinFile("mm3d Martini ") 
                       + "NKS-Set-OfFile@" + aName;


    if (aNamePrev=="")
        aCom += " OriOut=" + StdPrefix(aName);
    else
    {
       
        aCom += " InOri=" + StdPrefix(aInOri) +
                " OriOut=" + StdPrefix(aName);
    }


	if (EAMIsInit(&mInCal))	
		aCom += " OriCalib=" + mInCal;

	if (EAMIsInit(&mSH))	
		aCom += " SH=" + mSH;

	std::cout << "aCom=" << aCom << "\n";

	TopSystem(aCom.c_str());
	
}

/* Two options : 
   a- DoSBBA DoBASCULE=false - Martini with    InOri from prev sliding window to keep the repere
   b- DoSBBA DoBASCULE=true  - Martini without InOri but with Morito    
    then, LBA with Campari if desired */
void cAppliLuxor::DoSBBA()
{
	int i      =0;

    std::list<std::string>::iterator aW = mWName.begin();
    for ( ; aW!=mWName.end(); aW++)
    {
        if (i==0)
            DoSBBA(*aW);
        else if(DoBASCULE)
        {
            DoSBBA(*aW);
            
            if (LBA_ACTIVE)
                BasculeMorito (*aW,*std::prev(aW,1));
            else
            {
                BasculeMorito (*aW,*std::prev(aW,1));
            }
        }
        else
        {
            //if (LBA_ACTIVE)
                DoSBBA(*aW,*std::prev(aW,1));
            //else
			  //  DoSBBA(*aW,*std::prev(aW,i));
        }

        if (DoLBA)
        {
            if ( (i+1)%mLBAPas == 0)
            {
                LBA_ACTIVE = true;

                std::list<std::string> aLBAWAll;
                for (int aLBAWTmp=mLBAPas-1; aLBAWTmp>=0; aLBAWTmp--)
                {
                    if (ERupnik_MM())
                        std::cout << "ewwwwwwww="<< aLBAWTmp << " " << *std::prev(aW,aLBAWTmp) << "\n";
                    
                    aLBAWAll.push_back( *std::prev(aW,aLBAWTmp) );
                }

                LBACampari(aLBAWAll);
                
            }
            else
                LBA_ACTIVE = false;
        }
                
        i++;
    }
}





cAppliLuxor::cAppliLuxor(int argc, char** argv) :
	mWSz(5),
	mWPas(2),
	mLBAPas(2),
	mInOri(""),
	mInCal(""),
	mSH(""),
	mOut("Luxor"),
    mWinPrefix("Fen-"),
    mWinOriPostfix("-Ori"),
    DoBASCULE(false),
    DoLBA(false),
    LBA_ACTIVE(false)
{
	std::string aPattern;

	ElInitArgMain
	(
		argc, argv,
		LArgMain() << EAMC(mStrType,"Orientation mode (enum values)")
		           << EAMC(aPattern,"Pattern of images")
				   << EAMC(mWSz,"Processing window size"),
		LArgMain() << EAM (mWPas,"M",true,"Motion of the processing window in frames; Def=2")  
				   << EAM (mLBAPas,"LBAPas",true,"BBa step; Def=2 every other processing window")  
				   << EAM (mInOri,"InOri",true,"Input external orientation")  
				   << EAM (mInCal,"InCal",true,"Input internal orientation")  
				   << EAM (DoBASCULE,"Basc",true,"Do Bascule with Morito, Def=false")  
				   << EAM (DoLBA,"LBA",true,"Do LBA with Campari, Def=false")  
				   << EAM (mSH,"SH",true,"Homol prefix") 
				   << EAM (mOut,"Out",true,"Output orientation") 
	);

	
#if (ELISE_windows)
      replace( aPattern.begin(), aPattern.end(), '\\', '/' );
#endif
    SplitDirAndFile(mDir,mIms,aPattern);
    StdCorrecNameOrient(mInOri,mDir);

	ReadType(mStrType);


	mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    mSetIm = mICNM->Get(mIms);
    mNbIm = (int)mSetIm->size();


	mNbWinAll = std::floor(double((mNbIm-1) - (mWSz-1))/mWPas) +1;

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
			break;

		case eUndefVal :
			break;
	}
}

void cAppliLuxor::ReadType(const std::string & aType)
{
    mStrType = aType;
    StdReadEnum(mModeHelp,mType,mStrType,eUndefVal);
/*	eTypeMalt   xType;
    StdReadEnum(mModeHelp,xType,mStrType,eNbTypesMNE);*/
}


int Luxor_main(int argc,char** argv)
{
	cAppliLuxor anAppli(argc,argv);

    BanniereLuxor();

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
