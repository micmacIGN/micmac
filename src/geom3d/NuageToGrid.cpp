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



/***********************************************/
/*                                             */
/*             cMailageSphere                  */
/*                                             */
/***********************************************/


/***********************************************/
/*                                             */
/*                                             */
/*                                             */
/***********************************************/

static const INT TheNbMax= 4;

class cPtLaser
{
     public :
         cPtLaser (Pt3dr aP) :
                 mP (aP)
         {
         }


        Pt3dr mP;
        U_INT1 mCols[TheNbMax];
};

class cCmpPtYouZ
{
	public :

	   bool operator()(const cPtLaser & aP1, const cPtLaser & aP2)
	   {
                return (aP1.mP.y < aP2.mP.y) ;
	   }
	private:
};


class cLoadImDist
{
	public :
		cLoadImDist();
		void Load(int argc,char ** argv);

	private :
		void SauvTifFile(Fonc_Num,const std::string &,GenIm::type_el);
		// void InitClip();

	        Pt2dr FromPix(Pt2di aIndTP)   
		{
                       return mMSph.Pix2Spherik(Pt2dr(aIndTP)); 
		}
		Pt2di ToPix(REAL aTeta,REAL aPhi)
		{
                       return mMSph.Spherik2PixI(Pt2dr(aTeta,aPhi));
		}
		void AddTestStep(INT aK)
                {
		     if ( (mPts[aK].mP.z > mCurZ) && (mPts[aK].mP.z<minZSup))
                         minZSup = mPts[aK].mP.z;
                }


		// Pour Rectification

	       Pt3dr P3dOfPix(Pt2di anIndTetaPhi) ;
	       Pt3dr DirMailleOfPix(Pt2di anIndTetaPhi) ;
	       REAL ScoreInTriangle(Pt3dr & aRes,Pt3dr aDir,
			         Pt3dr aQ0,Pt3dr aP1,Pt3dr aP2);
	       Pt3dr RectifiedPoint(Pt2di anIndTetaPhi);
	       void RectifieImage();

		Pt2di       mSzW;
		Video_Win * pW;
		std::string  mPref;
		Im2D_INT1    mImCpt;
		Im2D_REAL4   mImX;
		Im2D_REAL4   mImY;
		Im2D_REAL4   mImZ;

		Im2D_REAL4   mImXRect;
		Im2D_REAL4   mImYRect;
		Im2D_REAL4   mImZRect;
	        ElMatrix<REAL> mMatrRect;

		Im2D_REAL4   mImProf;
		Im2D_REAL4   mImShade;
		std::vector<cPtLaser> mPts;
		INT                mNbP;
		cMailageSphere     mMSph;

		Pt2dr               mStep;
		Pt2di               mSzIm;

		//REAL               mT0;
		//REAL               mT1;
		//REAL               mP0;
		//REAL               mP1;
		REAL               mCurZ;
		REAL               minZSup;

                std::vector<INT>   mCanalAux;
                std::vector<Im2D_U_INT1 * >   mImAux;
                INT                mNbAux;
                INT                mCanalMax;
                double             mCols[TheNbMax];

                INT                mNbDir;
};


cLoadImDist::cLoadImDist () :
   mSzW   (800,800),
   pW     (Video_Win::PtrWStd(mSzW)),
   mImCpt  (1,1),
   mImX    (1,1),
   mImY    (1,1),
   mImZ    (1,1),
   mImXRect    (1,1),
   mImYRect    (1,1),
   mImZRect    (1,1),
   mMatrRect   (3,3),
   mImProf (1,1),
   mImShade (1,1),
   mMSph    (Pt2dr(0,0),Pt2dr(0,0),Pt2dr(0,0),true),
   mNbDir   (120)
{
}

Pt3dr cLoadImDist::P3dOfPix(Pt2di aTetaPhi) 
{
	return Pt3dr
		(
		    mImX.data()[aTetaPhi.y][aTetaPhi.x],
		    mImY.data()[aTetaPhi.y][aTetaPhi.x],
		    mImZ.data()[aTetaPhi.y][aTetaPhi.x]
		);
}

Pt3dr cLoadImDist::DirMailleOfPix(Pt2di anIndTetaPhi) 
{
    Pt2dr  aTP = FromPix(anIndTetaPhi);
    return Pt3dr::TyFromSpherique(1.0,aTP.x,aTP.y);
}

void cLoadImDist::RectifieImage()
{
    ELISE_COPY(mImX.all_pts(),mImX.in(),mImXRect.out());
    ELISE_COPY(mImY.all_pts(),mImY.in(),mImYRect.out());
    ELISE_COPY(mImZ.all_pts(),mImZ.in(),mImZRect.out());

    for (INT aX =1; aX<mSzIm.x-1 ; aX++)
    {
        if ((aX % 50) == 0)
            cout << "To Do = " << (mSzIm.x - aX) << "\n";
        for (INT aY =1; aY<mSzIm.y-1 ; aY++)
        {
		Pt3dr aP = RectifiedPoint(Pt2di(aX,aY));
		mImXRect.data()[aY][aX] = (float) aP.x;
		mImYRect.data()[aY][aX] = (float) aP.y;
		mImZRect.data()[aY][aX] = (float) aP.z;
        }
    }
}

Pt3dr cLoadImDist::RectifiedPoint(Pt2di anIndTetaPhi)
{
     for(INT aK=0 ; aK<9 ; aK++)
     {
         Pt2di aP = anIndTetaPhi + TAB_9_NEIGH[aK];
         if (mImCpt.data()[aP.y][aP.x] < 0)
             return P3dOfPix(anIndTetaPhi);
     }

     Pt3dr aDir = DirMailleOfPix(anIndTetaPhi);
     Pt3dr aQ0 = P3dOfPix(anIndTetaPhi);
     static Pt3dr VQ[5];
     for (INT aK=0 ; aK < 4 ; aK++)
         VQ[aK] = P3dOfPix(anIndTetaPhi+TAB_4_NEIGH[aK]);
     VQ[4] = VQ[0];

     REAL BestScore = - 1e20;
     static REAL  Score[4];
     Pt3dr aRes;
     for (INT aK=0 ; aK < 4 ; aK++)
     {
        Pt3dr anInter;
        Score[aK] = ScoreInTriangle(anInter,aDir,aQ0,VQ[aK],VQ[aK+1]);
	if ((aK==0) || (Score[aK]>BestScore))
	{
		BestScore = Score[aK];
		aRes = anInter;
	}
     }

     /*
     cout << anIndTetaPhi 
	  << " "<<  Score[0] << " "<<  Score[1]
	  << " "<<  Score[2] << " "<<  Score[3]
	  << "\n";
	  */

     return aRes;
}

REAL  cLoadImDist::ScoreInTriangle
      (
           Pt3dr & aRes,Pt3dr aDir,
           Pt3dr aQ0,Pt3dr aP1,Pt3dr aP2
)
{
    SetCol(mMatrRect,0,aDir);
    SetCol(mMatrRect,1,aP1-aQ0);
    SetCol(mMatrRect,2,aP2-aQ0);
    bool aOK = self_gaussj_svp(mMatrRect);

    if (! aOK)
    {
       cout << aDir << aQ0 << aP1 << aP2 << "\n";
       ELISE_ASSERT(false,"cLoadImDist::ScoreInTriangle");
    }

    Pt3dr ABC = mMatrRect * aQ0;

    aRes = aDir * ABC.x;

    return ElMin(ABC.y,ABC.z);
}



/*
void cLoadImDist::InitClip()
{
     cout << "Enter Borne Teta (entre 0 et 1 ) " << "\n";
     cin >> mT0;
     cin >> mT1;
     cout << "Enter Borne Phi (entre 0 et 1 ) " << "\n";
     cin >> mP0;
     cin >> mP1;

     mT0 = mTetaMin + (mTetaMax-mTetaMin) * mT0;
     mT1 = mTetaMin + (mTetaMax-mTetaMin) * mT1;
     mP0 = mPhiMin +  (mPhiMax-mPhiMin) * mP0;
     mP1 = mPhiMin +  (mPhiMax-mPhiMin) * mP1;
}
*/


void cLoadImDist::SauvTifFile
     (
          Fonc_Num aFonc,
	  const std::string & aNameShort,
	  GenIm::type_el  aType
     )
{
    std::string  aName = mPref + aNameShort + ".tif";

    Tiff_Im aFile
	    (
	        aName.c_str(),
		mSzIm,
		aType,
		Tiff_Im::No_Compr,
		(aFonc.dimf_out() == 1)?Tiff_Im::BlackIsZero : Tiff_Im::RGB
	    );

    ELISE_COPY(aFile.all_pts(),aFonc,aFile.out());
}


void cLoadImDist::Load(int argc,char ** argv)
{
    ELISE_ASSERT(argc>=2,"Bas Number of Arg in cLoadImDist::Load");

    std::string aNameFile;
    ElInitArgMain
    (
                argc,argv,
                LArgMain()      << EAM(aNameFile),
                LArgMain()  << EAM(mCanalAux,"CanalAux",true)
                            << EAM(mNbDir,"NbDir",true)
    );

    mNbAux = (int) mCanalAux.size();
    mCanalMax = -1;
    for (INT aK= 0 ; aK<mNbAux ; aK++)
    {
        ELISE_ASSERT
        (
            (mCanalAux[aK] >=0 ) && (mCanalAux[aK] <TheNbMax),
            "Bad Canal Aux in cLoadImDist::Load"
        );
        ElSetMax(mCanalMax,mCanalAux[aK]);
    }

    mPref = StdPrefix(aNameFile);
    FILE * aFP = ElFopen(aNameFile.c_str(),"r");
    ELISE_ASSERT(aFP!=0,"Cannot Open File for Laser Data");

    INT Nb=3;
    mNbP = 0;

    REAL aTetaMax = -1000;
    REAL aTetaMin = 1000;
    REAL aPhiMax = -1000;
    REAL aPhiMin = 1000;
    mPts.clear();

    char Buf[10000];
    while (Nb>=3)
    {
        Pt3dr  aP;

	Nb=0;
	char * got = fgets(Buf,10000,aFP);
	if (got)
	{
           Nb = sscanf(Buf,"%lf %lf %lf %lf %lf %lf %lf",
                           &aP.x,&aP.y,&aP.z,
                           &mCols[0],&mCols[1],&mCols[2],&mCols[3]
                );
	}
	if (Nb>=3)
	{
            ELISE_ASSERT((Nb-3) > mCanalMax,"Not Enough Chanel");
	    REAL Rho,Teta,Phi;
	    ToSpherique(aP,Rho,Teta,Phi);
	    ElSetMin(aTetaMin,Teta);
	    ElSetMax(aTetaMax,Teta);
	    ElSetMin(aPhiMin,Phi);
	    ElSetMax(aPhiMax,Phi);
	    mPts.push_back(cPtLaser(Pt3dr(Rho,Teta,Phi)));
            cPtLaser & aP = mPts.back();

            for (INT aK=0; aK <mNbAux ; aK++)
                aP.mCols[aK] = (U_INT1)mCols[mCanalAux[aK]];

	    mNbP++;
	    if ((mNbP%1000)==0) 
                cout << mNbP << "\n";

	}
    }
    mMSph.SetMax(Pt2dr(aTetaMax,aPhiMax));
    mMSph.SetMin(Pt2dr(aTetaMin,aPhiMin));
    cout << "TETA  in [ " << aTetaMin << " , " << aTetaMax << "]\n";
    cout << "PHI   in [ " << aPhiMin <<  " , " << aPhiMax << "]\n";
    ElFclose(aFP);

    mStep.x = sqrt((aTetaMax-aTetaMin) * (aPhiMax-aPhiMin) / mNbP);
    mStep.y = mStep.x;


    /*******************************************/
    /*                                         */
    /*   Calcul du pas en Phi et Teta          */
    /*                                         */
    /*******************************************/

     cout << "STEP IN = " << mStep  << "\n";
     INT NbTest = 500;
     cElStatErreur aStater(NbTest);
     for (INT aKTri=0 ; aKTri<2 ; aKTri++)
     {
             REAL & aStep =  (aKTri==1) ?  mStep.x : mStep.y;
	     aStater.Reset();
	     // bool TriTeta = (aK==0);
             cCmpPtYouZ aCmp;
             std::sort(mPts.begin(),mPts.end(),aCmp);

	     for (INT KTest = 0 ; KTest < NbTest ; KTest ++)
	     {
		  INT aKP = ElMin(mNbP-1,round_ni(mNbP*NRrandom3()));
	          REAL  YMin = mPts[aKP].mP.y - aStep/2.0;
	          REAL  YMax = mPts[aKP].mP.y + aStep/2.0;

		  mCurZ = mPts[aKP].mP.z;
		  minZSup = mCurZ + 200 * aStep;
                  for 
                  (  
                        INT aKP1 = aKP-1; 
                        (aKP1>=0) &&  (mPts[aKP1].mP.y > YMin) ; 
                        aKP1--
                  )
                  {
			  AddTestStep(aKP1);
                  }
                  for 
                  (  
                        INT aKP1 = aKP+1; 
                        (aKP1<mNbP) &&  (mPts[aKP1].mP.y < YMax) ; 
                        aKP1++
                  )
                  {
			  AddTestStep(aKP1);
                  }

		  REAL aVal = minZSup - mCurZ;
		  if (aVal < 1.5 * aStep)
		     aStater.AddErreur(aVal);

	     }


	     aStep = aStater.Erreur(0.5);

	     for (INT aKP = 0 ; aKP < mNbP ; aKP++)
	     {
		     ElSwap(mPts[aKP].mP.y,mPts[aKP].mP.z);
	     }
     }
     mMSph.SetStep(mStep);
     cout << "STEP = " << mStep << "\n";
     // mStep = mStep * 0.9;

     mSzIm = mMSph.SZEnglob();
     cout << "Sz Im = " << mSzIm << "\n";

    /*******************************************/
    /*                                         */
    /*   Calcul de X,Y,Z en geomtrie Teta-Phi  */
    /*                                         */
    /*******************************************/
     mImCpt.Resize(mSzIm);
     mImX.Resize(mSzIm);
     mImY.Resize(mSzIm);
     mImZ.Resize(mSzIm);
     mImXRect.Resize(mSzIm);
     mImYRect.Resize(mSzIm);
     mImZRect.Resize(mSzIm);
     mImProf.Resize(mSzIm);
     mImShade.Resize(mSzIm);

     for (INT aK=0 ; aK< mNbAux ; aK++)
     {
        mImAux.push_back(new  Im2D_U_INT1(mSzIm.x,mSzIm.y,0));
     }

     ELISE_COPY(mImShade.all_pts(),0.0,mImShade.out());
     ELISE_COPY(mImCpt.all_pts(),-1,mImCpt.out());
     ELISE_COPY(mImX.all_pts(),1e5,mImX.out()|mImY.out()|mImZ.out());
     INT aNbDoublon = 0;
     for (INT aK=0 ; aK< INT(mPts.size()) ; aK++)
     {
	  Pt2di aPix = ToPix(mPts[aK].mP.y,mPts[aK].mP.z);
	  INT1 & aCpt = mImCpt.data()[aPix.y][aPix.x];

	  if (aCpt < 100)
	  {
	     if (aCpt == -1) 
	         aCpt = 0;
	     REAL4 & dX =  mImX.data()[aPix.y][aPix.x];
	     REAL4 & dY =  mImY.data()[aPix.y][aPix.x];
	     REAL4 & dZ =  mImZ.data()[aPix.y][aPix.x];
	     Pt3dr aP = Pt3dr::TyFromSpherique(mPts[aK].mP.x,mPts[aK].mP.y,mPts[aK].mP.z);
	     aP = (aP + Pt3dr(dX,dY,dZ) * REAL(aCpt)) / (aCpt+1.0);
	     dX = (float) aP.x;
	     dY = (float) aP.y;
	     dZ = (float) aP.z;
             for (INT aC=0 ; aC< mNbAux ; aC++)
             {
                 U_INT1 & aV  = mImAux[aC]->data()[aPix.y][aPix.x];
                 aV = (U_INT1) ((mPts[aK].mCols[aC] +  aV *  REAL(aCpt)) / (aCpt+1.0));
             }
	     aCpt++;
	     if (aCpt == 2) 
	        aNbDoublon++;
	  }
     }

    /*******************************************/
    /*                                         */
    /*   Bouchage des trou                     */
    /*                                         */
    /*******************************************/
     for (INT aKIter = 0 ; aKIter < 5 ; aKIter ++)
     {
          Neigh_Rel aV4 (Neighbourhood::v4());
          Neigh_Rel aV8 (Neighbourhood::v8());
	  Liste_Pts_INT4 aList(2);
          ELISE_COPY
          (
              select
	      (
	         select(mImCpt.all_pts(),mImCpt.in()<0),
		 aV4.red_max(mImCpt.in(-1)) >= 0
	      ),
	      0,
	      aList
	  );

	  Fonc_Num XYZ = Virgule(mImX.in(0),mImY.in(0),mImZ.in(0));
	  Fonc_Num fRho = sqrt
		         (
                               ElSquare(mImX.in_proj()) 
                             + ElSquare(mImY.in_proj()) 
                             + ElSquare(mImZ.in_proj())
                         );
	  Fonc_Num fPds = (1+mImCpt.in(-1));
	  Fonc_Num fRhoInt = aV8.red_sum (fRho*fPds)/aV8.red_sum(fPds);
	  Fonc_Num fTeta =  aTetaMax - mStep.x* FX;
	  Fonc_Num fPhi =   aPhiMax - mStep.y* FY;


          for (INT aK=0 ; aK< mNbAux ; aK++)
          {
               Im2D_U_INT1 anI = *(mImAux[aK]);
               ELISE_COPY
               (
                   aList.all_pts(),
                   aV8.red_sum (anI.in_proj()*fPds)/aV8.red_sum(fPds),
	           anI.out()
	       );
          }

          ELISE_COPY
          (
              aList.all_pts(),
	      Virgule
	      (
                   fRhoInt * cos(fPhi) * cos(fTeta),
                   fRhoInt * cos(fPhi) * sin(fTeta),
                   fRhoInt * sin(fPhi) 
	      ),
	      Virgule(mImX.out(),mImY.out(),mImZ.out())
	  );
	  ELISE_COPY(aList.all_pts(),0,mImCpt.out());

     }

     cout << "DOUBLON = " << aNbDoublon << "\n";
     ELISE_COPY(mImCpt.all_pts(),128,pW->ogray());
     ELISE_COPY
     (
         select(mImCpt.all_pts(),mImCpt.in()>=0),
	 mImY.in(),// mImDist.in(),
	 pW->ocirc()
     );

    /*******************************************/
    /*                                         */
    /*   Image d'eclairage                     */
    /*                                         */
    /*******************************************/

     Fonc_Num fProf = Square(mImX.in(0))+Square(mImY.in(0)) ; // +Square(mImZ.in(0))
     if (true)
     {
        RectifieImage();
        fProf = Square(mImXRect.in(0))+Square(mImYRect.in(0)) ; 
     }

     REAL PMax,PMin;
     ELISE_COPY
     (
         mImProf.all_pts(),
	 (-log(fProf)) / euclid(mStep),
	 mImProf.out() | VMax(PMax) | VMin(PMin)
     );

     cout << "PROF in [" << PMin << " , " << PMax << "]\n";


     for (INT kFiltre =0 ; kFiltre < 2 ; kFiltre++)
     {
	// REAL Mult = -100;
        ELISE_COPY
        (
              mImProf.all_pts(),
	      MedianBySort(mImProf.in_proj(),2),
	      // rect_median(mImProf.in_proj() * Mult,2,INT((PMin-3)*Mult)) / Mult,
	      mImProf.out() | (pW->odisc() << (int) P8COL::green)
        );
     }



     REAL SPds = 0;
     for (int i=0; i<mNbDir; i++)
     {
          cout << "Dir " << i << " Sur " << mNbDir << "\n";
	  REAL Teta  = (2*PI*i) / mNbDir ;
	  Pt2dr U(cos(Teta),sin(Teta));
          Pt2di aDir = Pt2di(U * (mNbDir * 4.0));
	  REAL Pds = 3 - euclid(U,Pt2dr(0,1));
	  Symb_FNum Gr =    (PI/2-atan(gray_level_shading(mImProf.in()))) 
		          * (255.0/ (PI/2));
	  SPds  += Pds;
          ELISE_COPY
          (
               line_map_rect(aDir,Pt2di(0,0),mImX.sz()),
	       mImShade.in()+Pds*Gr,
	       mImShade.out() | (pW->ogray() << (mImShade.in() / SPds))
	  );
     };

     ELISE_COPY(mImShade.all_pts(),mImShade.in()/SPds,mImShade.out());


     bool Color = false;

     Fonc_Num fShade =  mImShade.in();
     if (Color)
        fShade =  its_to_rgb(Virgule(fShade,mImProf.in(), 255));
     else
     {
	     REAL Pds = 1.0;
	     REAL Gama = 1.3;

	     fShade = fShade * Pds + 128 * (1+cos(mImProf.in() / 70.0)) * (1-Pds);

	     if (Gama != 1.0)
                fShade = 255.0 * pow(fShade/255,1/Gama);

	     fShade =   (mImCpt.in() >=0) * fShade
		      + (mImCpt.in() < 0) *  (((FX/4)+FY/8) %2) * 255;

     }

     

     ELISE_COPY(mImShade.all_pts(),fShade,Color ? pW->orgb() : pW->ogray());

     SauvTifFile (Max(1,fShade) * (FX!=0),"Shade",GenIm::u_int1);
     SauvTifFile (mImX.in(),"X",GenIm::real4);
     SauvTifFile (mImY.in(),"Y",GenIm::real4);
     SauvTifFile (mImZ.in(),"Z",GenIm::real4);
     SauvTifFile (mImCpt.in(),"Cpt",GenIm::int1);

     for (INT aK=0 ; aK< mNbAux ; aK++)
     {
           SauvTifFile (mImAux[aK]->in(),"Aux"+ ToString(aK),GenIm::u_int1);
     }
     mMSph.WriteFile(mPref+".mtd");
     cMailageSphere::FromFile(mPref+".mtd");
}

void TestImDist(int argc,char ** argv)
{

    // std::string aNameFile ("/data1/Laser/Station_1.neu");
    // std::string aNameFile ("/data1/Laser/test.neu");
    // std::string aNameFile ("/data1/Laser/grotte_Station1.neu");
      // std::string aNameFile ("grotte_station5.neu");

    cLoadImDist aLoader;
    aLoader.Load(argc,argv);

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
