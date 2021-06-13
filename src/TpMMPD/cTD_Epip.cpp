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
class cOneImTDEpip;
class cAppliTDEpip;

static const double TheDefCorrel = -2.0;

class cOneImTDEpip
{
	public :
	   cOneImTDEpip(const std::string &,cAppliTDEpip &);
	   
       std::string ComCreateImDownScale(double aScale) const;
       std::string NameFileDownScale(double aScale) const;
	   
	   const std::string &  mMyName;
	   cAppliTDEpip &      mAppli;
	   Tiff_Im               mTifIm;
	   std::string           mNameCam;
	   ElCamera *            mCam;
	
};

class cCelP2D_TDEpi
{
	public :
	   ///================================================
	   ///===== PRE-REQUIREMENT FOR 2D PROG DYN , transfer
	   ///===== from 3D volume to buffer of computation 
	   ///================================================
	    void InitTmp(const cTplCelNapPrgDyn<cCelP2D_TDEpi> & aCel)
        {
            *this = aCel.ArgAux();
        }
	private :
};

class cLoaedImTDEpip
{
	 public :
	 
	   ///================================================
	   ///===== PRE-REQUIREMENT FOR 2D PROG DYN ==========
	   ///================================================
	   	    
            typedef  cCelP2D_TDEpi tArgCelTmp;
            typedef  cCelP2D_TDEpi tArgNappe;
            
            ///-------- Not a PRE-REQUIREMENT but Helpull
       
            typedef  cTplCelNapPrgDyn<tArgNappe>    tCelNap;
            typedef  cTplCelOptProgDyn<tArgCelTmp>  tCelOpt;	 
            
		  ///-------- REQUIREMENT : kernell of computation
           void DoConnexion
           (
                  const Pt2di & aPIn, const Pt2di & aPOut,
                  ePrgSens aSens,int aRab,int aMul,
                  tCelOpt*Input,int aInZMin,int aInZMax,
                  tCelOpt*Ouput,int aOutZMin,int aOutZMax
           );
           
           /// Required : but do generally nothing
           
           void GlobInitDir(cProg2DOptimiser<cLoaedImTDEpip> &);      	
	 
	       ///===============  Support to PrgD
	       
	        static const int DynPrD = 1000;
	        int ToICost (double aCost) {return round_ni(DynPrD*aCost);}
	     
	   ///===============
	   	   
	   cLoaedImTDEpip(cOneImTDEpip &,double aScale,int aSzW);

       double CrossCorrelation(const Pt2di & aPIm1,int aPx,const cLoaedImTDEpip & aIm2,int aSzW);
       double Covariance(const Pt2di& aPIm1,int aPx,const cLoaedImTDEpip & aIm2,int aSzW);

       bool InsideW(const Pt2di & aPIm1,const Pt2di & aSzW) const
       {
		   
		   // std::cout << aPIm1 << " " << mTMIn.get(aPIm1,0) << "\n";
		   return mTMIn.get(aPIm1,0) != 0;
		   /*
          return     (aPIm1.x >= aSzW.x) 
                  &&  (aPIm1.y >= aSzW.y) 
                  &&  (aPIm1.x < mSz.x-aSzW.x)
                  &&  (aPIm1.y < mSz.y-aSzW.y)
                  &&   mTMIn.get(aPIm1,0) ;
            */
       }
       bool InsideW(const Pt2di & aPIm1,const int & aSzW) const
       {
           return InsideW(aPIm1,Pt2di(aSzW,aSzW));
       }
       
       void ComputePx( cLoaedImTDEpip &,
                        TIm2D<INT2,INT>    mTEnvInf,
                        TIm2D<INT2,INT>    mTEnvSup,
                        int aSzW
                      );
                      
       void ComputePx( cLoaedImTDEpip &,
                        int  aIntPx,
                        int aSzW
                      );
        bool In3DMasq(const Pt2di &aPt,int aPx,cLoaedImTDEpip & aI2);
        
	   
	   cOneImTDEpip & mOIE;
	   cAppliTDEpip &      mAppli;
	   std::string mNameIm;
	   Tiff_Im mTifIm;
	   Pt2di mSz;
	   TIm2D<REAL4,REAL8> mTIm;
	   Im2D<REAL4,REAL8>  mIm;
	   TIm2D<REAL4,REAL8> mTImS1;
	   Im2D<REAL4,REAL8>  mImS1;
	   TIm2D<REAL4,REAL8> mTImS2;
	   Im2D<REAL4,REAL8>  mImS2;
	   
	   TIm2D<INT2,INT>    mTPx;
	   Im2D<INT2,INT>     mIPx;
	   TIm2D<U_INT1,INT>  mTSc;
	   Im2D<U_INT1,INT>   mISc;
	 
	   Im2D_Bits<1>       mMasqIn;
	   TIm2DBits<1>       mTMIn;  
	   Im2D_Bits<1>       mMasqOut;
	   TIm2DBits<1>       mTMOut;
	   int                mMaxJumpPax;
	   double              mRegul;
	   
	   
};






class cAppliTDEpip
{
	 public :
	    cAppliTDEpip(int argc, char **argv);
	     void GenerateDownScale(int aZoomBegin,int aZoomEnd);
	     
	     void DoMatchOneScale(int aZoomBegin,int aSzW);
	
	     std::string mNameIm1;
	     std::string mNameIm2;
	     std::string mDir;
	     cInterfChantierNameManipulateur * mICNM;
	     
	     std::string mNameMasq3D;
	     cMasqBin3D *  mMasq3D;
	     cOneImTDEpip * mIm1;
	     cOneImTDEpip * mIm2;
	     int             mZoomDeb;
	     int             mZoomEnd;
	     double         mRatioIntPx;
	     double         mIntPx;
	     int            mCurZoom;

};

/*************************************************/
/***        cAppliTDEpip                       ***/
/*************************************************/

cAppliTDEpip::cAppliTDEpip(int argc, char **argv) :
    mMasq3D   (0),
    mZoomDeb  (16),
    mZoomEnd  (2),
    mRatioIntPx (0.2)
{
	Pt2di aZoom;
    ElInitArgMain
    (
       argc,argv,
       LArgMain()  << EAMC(mNameIm1, "Firt Epip Image",eSAM_IsExistFile)
                   << EAMC(mNameIm2,"Second Epip Image",eSAM_IsExistFile),
       LArgMain()  << EAM(mNameMasq3D,"Masq3D",true,"3 D Optional masq")
                   << EAM(aZoom,"Zoom",true,"[ZoomBegin,ZoomEnd], Def=[16,2]")
    );

    if (EAMIsInit(&aZoom))
    {
		mZoomDeb = aZoom.x;
		mZoomEnd = aZoom.y;
    }
 
      mDir = DirOfFile(mNameIm1);
      mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
      
      if (EAMIsInit(&mNameMasq3D))
         mMasq3D = cMasqBin3D::FromSaisieMasq3d(mNameMasq3D);
         
      mIm1 = new cOneImTDEpip(mNameIm1,*this);
      mIm2 = new cOneImTDEpip(mNameIm2,*this);

     mIntPx =   mIm1->mCam->SzPixel().x * mRatioIntPx;
     
	GenerateDownScale(mZoomDeb,mZoomEnd);
	
	DoMatchOneScale(mZoomDeb,2);
}

void cAppliTDEpip::GenerateDownScale(int aZoomBegin,int aZoomEnd)
{
	std::list<std::string> aLCom;
	for (int aZoom = aZoomBegin ; aZoom >= aZoomEnd ; aZoom /=2)
	{
		std::string aCom1 = mIm1->ComCreateImDownScale(aZoom);
		std::string aCom2 = mIm2->ComCreateImDownScale(aZoom);
		if (aCom1!="") aLCom.push_back(aCom1);
		if (aCom2!="") aLCom.push_back(aCom2);
    }
    		
   cEl_GPAO::DoComInParal(aLCom);
}

void cAppliTDEpip::DoMatchOneScale(int aZoom,int aSzW) 
{
	mCurZoom = aZoom;
	cLoaedImTDEpip aLIm1(*mIm1,aZoom,aSzW);
	cLoaedImTDEpip aLIm2(*mIm2,aZoom,aSzW);
	
	aLIm1.ComputePx(aLIm2,round_up(mIntPx/aZoom) ,aSzW);
     
}


/*************************************************/
/***        cOneImTDEpip                       ***/
/*************************************************/


cOneImTDEpip::cOneImTDEpip(const std::string & aName,cAppliTDEpip & anAppli) :
   mMyName (aName),
   mAppli  (anAppli),
   mTifIm  (mMyName.c_str()),
   mNameCam (mAppli.mICNM->Assoc1To1("NKS-Assoc-Im2Orient@-Epi",mMyName,true)),
   mCam     (Cam_Gen_From_File(mNameCam,"OrientationConique",mAppli.mICNM))
{
}


std::string cOneImTDEpip::NameFileDownScale(double aScale) const
{
	return mAppli.mDir + "Tmp-MM-Dir/Scaled-" + ToString(aScale) + "-" +mMyName;
}


std::string cOneImTDEpip::ComCreateImDownScale(double aScale) const
{
    std::string aNameRes = NameFileDownScale(aScale);
	
    if (ELISE_fp::exist_file(aNameRes)) return "";
    
    return    MM3dBinFile("ScaleIm ") 
            +  mMyName
            +  " "  + ToString(aScale)
            +  " Out="  + aNameRes;
}

/*************************************************/
/***        cLoaedIm                      	   ***/
/*************************************************/

cLoaedImTDEpip::cLoaedImTDEpip(cOneImTDEpip & aOIE,double aScale,int aSzW) :
  mOIE (aOIE),
  mAppli (aOIE.mAppli),
  mNameIm (aOIE.NameFileDownScale(aScale)),
  mTifIm  (mNameIm.c_str()),
  mSz     (mTifIm.sz()),
  mTIm    (mSz),
  mIm     (mTIm._the_im),
  mTImS1  (mSz),
  mImS1   (mTImS1._the_im),
  mTImS2  (mSz),
  mImS2   (mTImS2._the_im),
  mTPx    (mSz),
  mIPx    (mTPx._the_im),
  mTSc    (mSz),
  mISc    (mTSc._the_im),
  mMasqIn (mSz.x,mSz.y,1),
  mTMIn   (mMasqIn),
  mMasqOut(mSz.x,mSz.y,0),
  mTMOut  (mMasqOut),
  mMaxJumpPax (2),
  mRegul      (0.05)
{
	ELISE_COPY(mIm.all_pts(), mTifIm.in(),mIm.out());
	
	ELISE_COPY(mMasqIn.border(aSzW),0 ,mMasqIn.out());
	
	ELISE_COPY
	(
	
	    mIm.all_pts(),
	    rect_som(mIm.in_proj(),aSzW) / ElSquare(1+2*aSzW),
	    mImS1.out()
	);
	
	ELISE_COPY
	(
	    mIm.all_pts(),
	    rect_som(Square(mIm.in_proj()),aSzW) / ElSquare(1+2*aSzW),
	    mImS2.out()
	);
	
/**	
	ELISE_COPY
	(
	    mIm.all_pts(),
	    Min(255,10 *sqrt(Max(0,mImS2.in() - Square( mImS1.in()) ))) ,
	    mTifIm.out()
	);
	**/
	
/**	ELISE_COPY
	(
	    mIm.all_pts(), 
	    rect_max(255 - mIm.in_proj(),10), 
	    mTifIm.out()
	 );**/
}

double cLoaedImTDEpip::CrossCorrelation
        (
            const Pt2di & aPIm1,
            int aPx,
            const cLoaedImTDEpip & aIm2,
            int aSzW
        )
{
   if (! InsideW(aPIm1,aSzW)) return TheDefCorrel;
   
   Pt2di aPIm2 = aPIm1 + Pt2di(aPx,0);
   if (! aIm2.InsideW(aPIm2,aSzW)) return TheDefCorrel;
   
   double aS1 = mTImS1.get(aPIm1);
   double aS2 = aIm2.mTImS1.get(aPIm2);

   
   double aCov = Covariance(aPIm1,aPx,aIm2,aSzW)  -aS1*aS2;

   double aVar11 = mTImS2.get(aPIm1) - ElSquare(aS1);
   double aVar22 = aIm2.mTImS2.get(aPIm2) - ElSquare(aS2);
   
   return aCov / sqrt(ElMax(1e-5,aVar11*aVar22));
}

double cLoaedImTDEpip::Covariance
       (
             const Pt2di & aPIm1,
             int aPx,
             const cLoaedImTDEpip & aIm2,
             int aSzW
       )
{
    if (1) // A test to check the low level access to data
    {
        float ** aRaw2 = mIm.data();
        float *  aRaw1 = mIm.data_lin();
        ELISE_ASSERT(mTIm.get(aPIm1)==aRaw2[aPIm1.y][aPIm1.x],"iiiii");
        ELISE_ASSERT((aRaw1+aPIm1.y*mSz.x) ==aRaw2[aPIm1.y],"iiiii");
    }
    double aSom =0;
    Pt2di aPIm2 = aPIm1+Pt2di(aPx,0);

    Pt2di aVois;
    for (aVois.x=-aSzW; aVois.x<=aSzW ; aVois.x++)
    {
        for (aVois.y=-aSzW; aVois.y<=aSzW ; aVois.y++)
        {
             aSom +=  mTIm.get(aPIm1+aVois) * aIm2.mTIm.get(aPIm2+aVois);
        } 
    }
    return aSom /ElSquare(1+2*aSzW);
}

void cLoaedImTDEpip::ComputePx
      ( 
           cLoaedImTDEpip & aIm2,
           INT aPxMax,
           int aSzW
      )
{
	TIm2D<INT2,INT> aTEnvInf(mSz);
	TIm2D<INT2,INT> aTEnvSup(mSz);
	
	ELISE_COPY(aTEnvInf._the_im.all_pts(),-aPxMax,aTEnvInf._the_im.out());
	ELISE_COPY(aTEnvSup._the_im.all_pts(),1+aPxMax,aTEnvSup._the_im.out());

    ComputePx(aIm2,aTEnvInf,aTEnvSup,aSzW);
}

void cLoaedImTDEpip::DoConnexion
     (
                  const Pt2di & aPIn, const Pt2di & aPOut,
                  ePrgSens aSens,int aRab,int aMul,
                  tCelOpt*Input,int aInZMin,int aInZMax,
                  tCelOpt*Output,int aOutZMin,int aOutZMax
      )
{
	for (int aZ = aOutZMin ; aZ < aOutZMax ; aZ++)
	{
		int aDZMin,aDZMax;
		ComputeIntervaleDelta
		(
		    aDZMin,aDZMax,aZ,mMaxJumpPax,
		    aOutZMin,aOutZMax,
		    aInZMin,aInZMax
		);
               for (int aDZ = aDZMin; aDZ<= aDZMax ; aDZ++)
               {
			double aCost = mRegul * ElAbs(aDZ);
			Output[aZ].UpdateCostOneArc(Input[aZ+aDZ],aSens,ToICost(aCost));
	       }
    }
}

template <class  Type> class cMyAveragin : public Simple_OPBuf1<Type,Type>
{
	public :
	
	void  calc_buf (Type ** MCoutput,Type *** MCinput)
	{
		for (int aNbC=0 ; aNbC<this->dim_in() ; aNbC++)
		{
			Type * ImOut = MCoutput[aNbC]; 
			Type ** ImIn = MCinput[aNbC];
			
			for (int x=this->x0() ; x<this->x1() ; x++)
			{
				Type aSom = 0;
				for (int aDX=-mNbV ; aDX <=mNbV ; aDX++)
				{
					for (int aDY=-mNbV ; aDY <=mNbV ; aDY++)
				    {
                        aSom += ImIn[aDY][x+aDX];					
				    }
				}
				ImOut[x] = aSom;
			}
		}
	}
	
	cMyAveragin(int aNbV) : mNbV (aNbV) {}
	int mNbV;
	
};
Fonc_Num MySom(Fonc_Num f,int aNbV)
{
    return create_op_buf_simple_tpl
           (
                new   cMyAveragin<int>(aNbV),
                new   cMyAveragin<double>(aNbV),
                f,
                1,
                Box2di (Pt2di(-aNbV,-aNbV), Pt2di(aNbV,aNbV))
           );
}



void cLoaedImTDEpip::GlobInitDir(cProg2DOptimiser<cLoaedImTDEpip> &)
{
}

void cLoaedImTDEpip::ComputePx
      ( 
           cLoaedImTDEpip & aIm2,
           TIm2D<INT2,INT>    aTEnvInf,
           TIm2D<INT2,INT>    aTEnvSup,
           int aSzW
      )
{
	
	if (0)
	{
		Video_Win aW = Video_Win::WStd(mSz,2);
		
		Fonc_Num aF = mIm.in(0);
		for (int aK=0 ; aK<4 ; aK++)
		    aF = MySom(aF,2) / 25;
		
		ELISE_COPY
		(
		    mIm.all_pts(),
		    rect_min(rect_max(255-aF,3),3),
		    aW.ogray()
		);
		aW.clik_in();
		    
    }
    
	Pt2di aP; 
	///  PRGD 1 : create the object
	
	cProg2DOptimiser<cLoaedImTDEpip> aPrgD(*this,aTEnvInf._the_im,aTEnvSup._the_im,0,1);
    cDynTplNappe3D<tCelNap> &  aSparseVol = aPrgD.Nappe();
    tCelNap ***  aSparsPtr = aSparseVol.Data() ;
    ///  -- end PRGD 1

	for (aP.x =0 ; aP.x < mSz.x ; aP.x++)
	{
	   for (aP.y =0 ; aP.y < mSz.y ; aP.y++)
	   {
		   int aPxMin = aTEnvInf.get(aP);
		   int aPxMax = aTEnvSup.get(aP);
		   int aBestPax = 0;
		   double aBestCor = TheDefCorrel;
		   
		   for (int aPax = aPxMin ; aPax < aPxMax ; aPax++)
		   {
			   double aCor = TheDefCorrel;
			   if (In3DMasq(aP,aPax,aIm2))
			   {
	 		      aCor = CrossCorrelation(aP,aPax,aIm2,aSzW);
			      if (aCor > aBestCor)
			      {
				     aBestCor = aCor;
				     aBestPax = aPax;
			      }
		      }
		      ///  PRGD 2 : fill the cost 
		      aSparsPtr[aP.y][aP.x][aPax].SetOwnCost(ToICost(1-aCor));
		      /// == End PGRD2
           }	
           mTPx.oset(aP,aBestPax);	
           mTSc.oset(aP,ElMax(0,ElMin(255,round_ni((aBestCor+1)*128))));
	   }
	}
	
	/// PRGD3 : run the optim and use the result
	aPrgD.DoOptim(7);
	Im2D_INT2 aSolPrgd(mSz.x,mSz.y);
    aPrgD.TranfereSol(aSolPrgd.data());
    Tiff_Im::CreateFromIm(aSolPrgd,"TestPrgPx.tif");
	///  end PRGD3
	
	
	
	if (1)
	{
		Video_Win aW = Video_Win::WStd(mSz,3);
		
		ELISE_COPY
		(
		    mTPx._the_im.all_pts(),
		    Min(2,Abs(mTPx._the_im.in()-aSolPrgd.in())),
		    aW.odisc()
		);
		aW.clik_in();
		    
    }
	
	Tiff_Im::CreateFromIm(mTPx._the_im,"TestPx.tif");
	Tiff_Im::CreateFromIm(mTSc._the_im,"TestSc.tif");

	std::cout << "DONE PX\n";
}


bool cLoaedImTDEpip::In3DMasq(const Pt2di &aPt,int aPx,cLoaedImTDEpip & aI2)
{
	cMasqBin3D *  aMasq = mAppli.mMasq3D;
	if (!aMasq) return true;
	
	Pt2di aFullP1 = aPt * mAppli.mCurZoom;
	Pt2di aFullP2 = aFullP1 + Pt2di(aPx*mAppli.mCurZoom,0);
	
	Pt3dr  aPGr =  mOIE.mCam->PseudoInter
	               (
	                  Pt2dr(aFullP1),
	                  *aI2.mOIE.mCam,
	                  Pt2dr(aFullP2)
	                );
	
	return aMasq->IsInMasq(aPGr);
}                  
                      
/*************   MAIN *****************/

int TDEpip_main(int argc, char **argv)
{
	cAppliTDEpip anAppli(argc,argv);
    std::cout << "TDEpip_main \n";

    return EXIT_SUCCESS;
}





/** Footer-MicMac-eLiSe-25/06/2007

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
