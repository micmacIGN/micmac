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

#include "TpPPMD.h"

/*
void f (int a,int X=0);
void f (int X=0,int Y =1);
void f (int X=0,int a,int Y =1);
*/

const float Beaucoup=1e20;

bool  VignetteInImage(int aSzW,const cTD_Im & aIm1,const Pt2di & aP1)
{
	Pt2di aSzIm = aIm1.Sz();
	
	return      (aP1.x >= aSzW)
	          && (aP1.x < aSzIm.x - aSzW)
	          && (aP1.y >= aSzW)
			  && (aP1.y < aSzIm.y - aSzW);
/*
	aIm1.Ok(aP1.x-aSzW,aP1.y)
	        &&  aIm1.Ok(aP1.x+aSzW,aP1.y)
	        &&  aIm1.Ok(aP1.x,aP1.y-aSzW)
	        &&  aIm1.Ok(aP1.x,aP1.y+);
*/

}





float SimilByCorrel
      (
         int aSzW,
         const cTD_Im & aIm1,const Pt2di & aP1,
         const cTD_Im & aIm2,const Pt2di & aP2
      )
{
    if (! 	VignetteInImage(aSzW,aIm1,aP1)) return Beaucoup;
    if (! 	VignetteInImage(aSzW,aIm2,aP2)) return Beaucoup;

	RMat_Inertie aMat;
    
    Pt2di aDP;
    for (aDP.x= -aSzW ; aDP.x<= aSzW ; aDP.x++)
    {
        for (aDP.y= -aSzW ; aDP.y<= aSzW ; aDP.y++)
        {
			float aV1 = aIm1.GetVal(aP1+aDP);
			float aV2 = aIm2.GetVal(aP2+aDP);
			aMat.add_pt_en_place(aV1,aV2,1.0);
		}
	}

	return 1-aMat.correlation();
}

float SimilMultiW
      (
         const cTD_Im & aIm1,const Pt2di & aP1,
         const cTD_Im & aIm2,const Pt2di & aP2
      )
{
	
	float aS1 = SimilByCorrel(1,aIm1,aP1,aIm2,aP2);
    float aS2 = SimilByCorrel(2,aIm1,aP1,aIm2,aP2);
    float aS4 = SimilByCorrel(4,aIm1,aP1,aIm2,aP2);

	return std::max(aS1,std::max(aS2,aS4));
}




float SimilByDif
      (
         int aSzW,
         const cTD_Im & aIm1,const Pt2di & aP1,
         const cTD_Im & aIm2,const Pt2di & aP2
      )
{
    if (! 	VignetteInImage(aSzW,aIm1,aP1)) return Beaucoup;
    if (! 	VignetteInImage(aSzW,aIm2,aP2)) return Beaucoup;

    float aSom = 0;
    
    Pt2di aDP;
    for (aDP.x= -aSzW ; aDP.x<= aSzW ; aDP.x++)
        for (aDP.y= -aSzW ; aDP.y<= aSzW ; aDP.y++)
			aSom += fabs(aIm1.GetVal(aP1+aDP)-aIm2.GetVal(aP2+aDP));

	return aSom;
}

class cTD_OneScale
{
	public :
		const static int DefSzW=-1;
		
		cTD_OneScale() : // Uniquement pour pouvoir faire des vector
		   mIm (1,1)
		 {
		 }
		
         void FillInertie
		 (
			RMat_Inertie &aMat,
			const Pt2di & aP1,
			const cTD_OneScale & aIm2,const Pt2di & aP2
		 ) const
		 {
			Pt2di aDP;
			for (aDP.x= -mSzW ; aDP.x<= mSzW ; aDP.x+=mStepW)
			{
				for (aDP.y= -mSzW ; aDP.y<= mSzW ; aDP.y+=mStepW)
				{
					float aV1 = this->mIm.GetVal(aP1+aDP);
					float aV2 =   aIm2.mIm.GetVal(aP2+aDP);
					aMat.add_pt_en_place(aV1,aV2,mPdsByPix);
				}
			 }
		  }
		  
		  float SimilByCorrel
		  (
		    const Pt2di & aP1,
			const cTD_OneScale & aIm2,const Pt2di & aP2
		  ) const
		  {
			  RMat_Inertie aMat;
			  FillInertie(aMat,aP1,aIm2,aP2);
			  
			  return 1- aMat.correlation();
		  }
		  
		  //double Correl
	
		cTD_OneScale
		(
		     const cTD_Im & anIm,
		     double aPds,
		     int aScale,
		     int aNbIter,
		     int aSzW
		 ) :
		   mIm  (anIm),
		   mFullW (false),
		   mPds   (aPds),
		   mScale (aScale * std::sqrt(aNbIter)),
		   mSzW  ((aSzW==DefSzW) ? round_ni(aSzW) : aSzW),
		   mPdsByPix (mPds/( mFullW ? ElSquare(1+2*mSzW): 9)),
		   mStepW    (mFullW ? 1 :mSzW )
		{
			std::cout << "PDS = " << aPds << " SZW " << aSzW << "\n";
			if (aScale >0)
			   mIm = mIm.ImageMoy(aScale,aNbIter);
		}
	
		
		cTD_Im  mIm;
		bool   mFullW;
		double mPds;
		double mScale;
		int     mSzW;
		double mPdsByPix;
		int     mStepW;
};

class cTD_PyrMultiScale
{
	public :
	
		cTD_PyrMultiScale(const cTD_Im & aIm0) :
			mIm0    (aIm0),
			mSzWMax (0),
			mNbIm   (0),
			mSomPds (0)
		{
		}
		bool InPyram(const Pt2di & aP) const
		{
			return VignetteInImage(mSzWMax,mIm0,aP);
		}
		
		float SimilByCatCorrel
			  (
				const Pt2di & aP1,
				const cTD_PyrMultiScale & aPyr2,
				const Pt2di & aP2
				) const 
		{
			if ((!InPyram(aP1)) || (!aPyr2.InPyram(aP2)))
				return 2 ;
			
			RMat_Inertie aMat;
			for (int aK=0 ; aK<mNbIm ; aK++)
			{
				const cTD_OneScale&  aSC1 = mVecIms[aK];
				const cTD_OneScale&  aSC2 = aPyr2.mVecIms[aK];

				aSC1.FillInertie(aMat,aP1,aSC2,aP2);
			}
			
			return 1-aMat.correlation();
		}
		
		
		float SimilBySomCorrel
			  (
				const Pt2di & aP1,
				const cTD_PyrMultiScale & aPyr2,
				const Pt2di & aP2
				) const 
		{
			if ((!InPyram(aP1)) || (!aPyr2.InPyram(aP2)))
				return 2 ;
			double aRes = 0;
			
			for (int aK=0 ; aK<mNbIm ; aK++)
			{
				const cTD_OneScale&  aSC1 = mVecIms[aK];
				const cTD_OneScale&  aSC2 = aPyr2.mVecIms[aK];

				aRes += aSC1.mPds * aSC1.SimilByCorrel(aP1,aSC2,aP2);
			}
			
			return aRes / mSomPds;
		}
				
	
		
		void AddScale 
			(
				double aPds,
				int aScale,
				int aNbIter,
				int aSzW
			) 
		{
			mVecIms.push_back(cTD_OneScale(mIm0,aPds,aScale,aNbIter,aSzW));
			mNbIm++;
			mSzWMax = max(mSzWMax,mVecIms.back().mSzW);
			mSomPds += aPds;
		}
			   
		cTD_Im						 mIm0;
		std::vector<cTD_OneScale>  mVecIms;
		int							 mSzWMax;
		int							 mNbIm;
		double                      mSomPds;
};

class cPairPyramMS
{
	public :
		cPairPyramMS(const cTD_Im & aIm1,const cTD_Im & aIm2) :
			mPyr1 (aIm1),
			mPyr2 (aIm2)
		{
		}
		
		float SimilBySomCorrel
			  (
				const Pt2di & aP1,
				const Pt2di & aP2
			) const 
		{
			return mPyr1.SimilBySomCorrel(aP1,mPyr2,aP2);
		}
		
		float SimilByCatCorrel
			  (
				const Pt2di & aP1,
				const Pt2di & aP2
			) const 
		{
			return mPyr1.SimilByCatCorrel(aP1,mPyr2,aP2);
		}		
			
		void AddScale 
			(
				double aPds,
				int aScale,
				int aNbIter,
				int aSzW
			) 
		{
			mPyr1.AddScale(aPds,aScale,aNbIter,aSzW);
			mPyr2.AddScale(aPds,aScale,aNbIter,aSzW);	
		}
		cTD_PyrMultiScale mPyr1;
		cTD_PyrMultiScale mPyr2;

};



int  TD_Match3_main(int argc,char ** argv)
{
	bool ByCorrel = false;
    std::string aNameI1,aNameI2;
    int aDeltaPax=100;
    int aSzW = 5;
    
    
     ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameI1,"Name Im1")
					<< EAMC(aNameI2,"Name Im2"),
        LArgMain()  << EAM(aDeltaPax,"DPax",true,"Delta paralax")
                    << EAM(aSzW,"SzW",true,"Size of Window, Def=5")
                    << EAM(ByCorrel,"ByCorrel",true,"By correlation")
    );
    
     cTD_Im aI1 = cTD_Im::FromString(aNameI1);
     cTD_Im aI2 = cTD_Im::FromString(aNameI2);

	cPairPyramMS  aPPMS(aI1,aI2);
		
	// 	AddScale (Pds, int aScale,int aNbIter,int aSzW) 

	aPPMS.AddScale(4,0,1,1);
	aPPMS.AddScale(2,1,4,2);
	aPPMS.AddScale(1,2,4,4);
	aPPMS.AddScale(0.5,4,4,8);
	aPPMS.AddScale(0.25,8,4,16);



	

    
    Pt2di aP;
    // on charger nos deux images
    // image 1
    Pt2di aSz = aI1.Sz();
    std::string aNameMasqIm1 = std::string("Masq_") + aNameI1;
    cTD_Im aIMasq(aSz.x,aSz.y); 
    if (ELISE_fp::exist_file(aNameMasqIm1))
    {
		aIMasq = cTD_Im::FromString(aNameMasqIm1);
	}
	else
	{
	  for (aP.x=0; aP.x < aSz.x ; aP.x++)
      {
		for (aP.y=0 ; aP.y < aSz.y ; aP.y++)
		{
			aIMasq.SetVal(aP.x,aP.y,1);
		}
	  }
	}    
    
   

    
    // image 2
    
   //aI1 = aI1.ImageMoy(aSzW,1);
   // aI2 = aI2.ImageMoy(aSzW,1);
    
    // on crée un image pour stocker le résultat de la corrélation 
    cTD_Im aICorelMin = cTD_Im(aSz.x, aSz.y);
     // on crée la carte de profondeur
    cTD_Im aIProf = cTD_Im(aSz.x, aSz.y);
    
    for (aP.x=0; aP.x < aSz.x ; aP.x++)
    {
		if ((aP.x%50)==0) std::cout << "Reste " << aSz.x-aP.x << "\n";
		for (aP.y=0 ; aP.y < aSz.y ; aP.y++)
		{
		    float aSimMin = Beaucoup;
		    int aPaxOpt=0;
		    Pt2di aPPax(0,0);
		    if (aIMasq.GetVal(aP.x,aP.y))
		    {
				for ( aPPax.x = -aDeltaPax ; aPPax.x<=aDeltaPax ; aPPax.x++)
				{
					Pt2di aP2 = aP+aPPax;
					if (1)
					{
						// float aSimil =  aPPMS.SimilBySomCorrel(aP,aP2);
						float aSimil =  aPPMS.SimilByCatCorrel(aP,aP2);
	
									
						//float aDiff =	SimilMultiW(aI1,aP,aI2,aP2);
						if  (aSimil < aSimMin)
						{
							aSimMin = aSimil;
							aPaxOpt = aPPax.x;
						}
					}
				}
	    	} 
			aIProf.SetVal(aP.x,aP.y,aPaxOpt);
		}
	}
	
	std::string aNameRes = "CartePax";
	aNameRes += std::string("_SzW") + ToString(aSzW);
	aNameRes +=  ByCorrel ? "Correl" :  "Dif";
	aNameRes += ".tif";
		
    aIProf.Save(aNameRes );
    
    System("to8Bits " + aNameRes + " Circ=1 Dyn=10");
    
	return EXIT_SUCCESS;
}





// +Pt2di(3,4)

int  TD_Match1_main(int argc,char ** argv)
{
	bool ByCorrel = false;
    std::string aNameI1,aNameI2;
    int aDeltaPax=100;
    int aSzW = 5;
    
    
     ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameI1,"Name Im1")
					<< EAMC(aNameI2,"Name Im2"),
        LArgMain()  << EAM(aDeltaPax,"DPax",true,"Delta paralax")
                    << EAM(aSzW,"SzW",true,"Size of Window, Def=5")
                    << EAM(ByCorrel,"ByCorrel",true,"By correlation")
    );
    Pt2di aP;
    // on charger nos deux images
    // image 1
    cTD_Im aI1 = cTD_Im::FromString(aNameI1);
    Pt2di aSz = aI1.Sz();
    std::string aNameMasqIm1 = std::string("Masq_") + aNameI1;
    cTD_Im aIMasq(aSz.x,aSz.y); 
    if (ELISE_fp::exist_file(aNameMasqIm1))
    {
		aIMasq = cTD_Im::FromString(aNameMasqIm1);
	}
	else
	{
	  for (aP.x=0; aP.x < aSz.x ; aP.x++)
      {
		for (aP.y=0 ; aP.y < aSz.y ; aP.y++)
		{
			aIMasq.SetVal(aP.x,aP.y,1);
		}
	  }
	}    
    
   

    
    // image 2
    cTD_Im aI2 = cTD_Im::FromString(aNameI2);
    
   //aI1 = aI1.ImageMoy(aSzW,1);
   // aI2 = aI2.ImageMoy(aSzW,1);
    
    // on crée un image pour stocker le résultat de la corrélation 
    cTD_Im aICorelMin = cTD_Im(aSz.x, aSz.y);
     // on crée la carte de profondeur
    cTD_Im aIProf = cTD_Im(aSz.x, aSz.y);
    
    for (aP.x=0; aP.x < aSz.x ; aP.x++)
    {
		if ((aP.x%50)==0) std::cout << "Reste " << aSz.x-aP.x << "\n";
		for (aP.y=0 ; aP.y < aSz.y ; aP.y++)
		{
		    float aDiffMin = Beaucoup;
		    int aPaxOpt=0;
		    Pt2di aPPax(0,0);
		    if (aIMasq.GetVal(aP.x,aP.y))
		    {
				for ( aPPax.x = -aDeltaPax ; aPPax.x<=aDeltaPax ; aPPax.x++)
				{
					Pt2di aP2 = aP+aPPax;
					if (1)
					{
						float aDiff =  ByCorrel ?
									SimilByCorrel(aSzW,aI1,aP,aI2,aP2):			                
									SimilByDif(aSzW,aI1,aP,aI2,aP2);
									
						//float aDiff =	SimilMultiW(aI1,aP,aI2,aP2);
						if  (aDiff < aDiffMin)
						{
							aDiffMin = aDiff;
							aPaxOpt = aPPax.x;
						}
					}
				}
	    	} 
			aIProf.SetVal(aP.x,aP.y,aPaxOpt);
		}
	}
	
	std::string aNameRes = "CartePax";
	aNameRes += std::string("_SzW") + ToString(aSzW);
	aNameRes +=  ByCorrel ? "Correl" :  "Dif";
	aNameRes += ".tif";
		
    aIProf.Save(aNameRes );
    
    System("to8Bits " + aNameRes + " Circ=1 Dyn=10");
    
	return EXIT_SUCCESS;
}

int  TD_Match2_main(int argc,char ** argv)
{

    std::string aNameI1,aNameI2;
    int aDeltaPax=100;
    int aSzW = 5;
    
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameI1,"Name Im1")
					<< EAMC(aNameI2,"Name Im2"),
        LArgMain()  << EAM(aDeltaPax,"DPax",true,"Delta paralax")
                    << EAM(aSzW,"SzW",true,"Size of Window, Def=5")
    );
    
       // image 1
    cTD_Im aI1 = cTD_Im::FromString(aNameI1);
    // image 2
    cTD_Im aI2 = cTD_Im::FromString(aNameI2);
    
    //dimension de nos images, sera utile pour nos boucles
    Pt2di aSz = aI1.Sz();
    
    cTD_Im aIBestScore(aSz.x,aSz.y);
    cTD_Im aIBestPax(aSz.x,aSz.y);
    Pt2di aP;    

    for (aP.x=0; aP.x < aSz.x ; aP.x++)
    {
		for (aP.y=0 ; aP.y < aSz.y ; aP.y++)
		{
		   aIBestScore.SetVal(aP.x,aP.y,Beaucoup);
		   aIBestPax.SetVal(aP.x,aP.y,sin(aP.x)*30*sin(aP.y));
		}
	}
    
    Pt2di aPPax;
    for ( aPPax.x = -aDeltaPax ; aPPax.x<=aDeltaPax ; aPPax.x++)
	{
		std::cout << "Pax= " << aPPax.x << "\n";
		
	// Calculer images des valeurs absolue des difference trans
		cTD_Im aImDif(aSz.x,aSz.y);

		for (aP.x=0; aP.x < aSz.x ; aP.x++)
		{
			for (aP.y=0 ; aP.y < aSz.y ; aP.y++)
			{
				Pt2di aPTr = aP + aPPax;
				float aDif = 256;
				if (aI2.Ok(aPTr.x,aPTr.y))
				{
					 aDif = aI1.GetVal(aP) - aI2.GetVal(aPTr);
				}
				aImDif.SetVal(aP.x,aP.y,std::fabs(aDif));
			}
		}
		

		 //  Calculer l'image moyenne
		 
		 cTD_Im aImDifMoy = aImDif.ImageMoy(aSzW,1);

		 // Mettre a jour aIBestScore et aIBestPax
		 
		 
		for (aP.x=0; aP.x < aSz.x ; aP.x++)
		{
			for (aP.y=0 ; aP.y < aSz.y ; aP.y++)
			{
				float aDif =aImDifMoy.GetVal(aP);
				if (aDif<aIBestScore.GetVal(aP))
				{
					 aIBestScore.SetVal(aP.x,aP.y,aDif);
					 aIBestPax.SetVal(aP.x,aP.y,aPPax.x);
				}
			}
		}
	}
	aIBestPax.Save("CartePax2.tif");
	
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