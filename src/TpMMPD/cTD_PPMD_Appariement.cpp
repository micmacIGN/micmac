#include "StdAfx.h"
#include "TpPPMD.h"

/*************************************
 *
 *  Creation d'une class image : cImg
 *
*************************************/
class cImg
{
    public :
        cImg(int aX,int aY);        
        cImg(const cImg &);

        static cImg Read(const std::string& aName);
        void Save(const std::string& aName);
        
        double Get(const Pt2di aP) {return mTIm.get(aP);}
        void Set(const Pt2di aP,double aVal) {mTIm.oset(aP,aVal);}

        Pt2di Sz() {return mSz;}



        cImg & operator = (const cImg&);
    private :
        Im2D<double,double>  mIm; //lib MicMac
        TIm2D<double,double> mTIm; //lib 
        
        Pt2di mSz;
};

enum eCorMode
{
    eSSD,
    eCORR,
    eCENSUS
};

eCorMode Str2Enum(std::string& aMode)
{
	eCorMode aRes;

	if (aMode=="SSD")
		aRes = eSSD;
	else if (aMode=="CORR")
		aRes = eCORR;
	else if (aMode=="CENSUS")
		aRes = eCENSUS;
	else
		aRes = eSSD;

	return aRes;
}

/*****************************
 *
 * The matching class
 *
 * ***************************/

class cMEC
{
    public :

		//class constructor	
        cMEC(const cImg& aI1,const cImg& aI2,
             int aSzW,int aPxMax, eCorMode aMode);
        
		// perform matching of all pixels in mI1
        void DoAll();

		// different functions to calculate the correlation
		// between two windows
        double Cost(int X,int Y,int X2);
        double CostSSD(int X,int Y,int X2);
        double CostCORR(int X,int Y,int X2);
        double CostCENSUS(int X,int Y,int X2);
       
	   	// getter function of the parallax image	
        cImg Px() {return mPxIm;}

    private:
		// correlation mode
        eCorMode mMode;
         
		// the images that you want to match
        cImg mI1;
        cImg mI2;
        
		// image sizes
        Pt2di mSz1;
        Pt2di mSz2;
        
		// correlation window
        int mSzW;
		// the tested parallax
        int mPxMax;

		// the correlation image (result)
        cImg mCorIm;
		// the parallax/surface image (result)
        cImg mPxIm;

};

cMEC::cMEC(const cImg& aI1,const cImg& aI2,
             int aSzW,int aPxMax, eCorMode aMode) :
    mMode(aMode),
    mI1(aI1),
    mI2(aI2),
    mSz1(mI1.Sz()),
    mSz2(mI2.Sz()),
    mSzW(aSzW),
    mPxMax(aPxMax),
    mCorIm(mSz1.x,mSz1.y),
    mPxIm(mSz1.x,mSz1.y)
{
    DoAll();
}

/* TO DO */
void cMEC::DoAll()
{
        
    
}

double cMEC::Cost(int X,int Y,int X2)
{
    switch (mMode)
    {
        case eSSD : return CostSSD(X,Y,X2);
        case eCORR : return CostCORR(X,Y,X2);
        case eCENSUS : return CostCENSUS(X,Y,X2);
    }

    return 0.0;
}


/* TO DO */
double cMEC::CostSSD(int X,int Y,int X2)
{

    return 1.0;

}

/* TO DO */
double cMEC::CostCORR(int X,int Y,int X2)
{
	return 1.0;
}

/* TO DO */
double cMEC::CostCENSUS(int X,int Y,int X2)
{
	return 1.0;
}

void cImg::Save(const std::string& aName)
{
    Tiff_Im aTif(aName.c_str(),
                 mSz,
                 GenIm::real4,
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero);
    ELISE_COPY(aTif.all_pts(),mIm.in(),aTif.out());
}

cImg cImg::Read(const std::string& aName)
{
    Tiff_Im aTif = Tiff_Im::StdConvGen(aName,-1,true);
    cImg aRes (aTif.sz().x,aTif.sz().y);

    ELISE_COPY(aTif.all_pts(),aTif.in_proj(),aRes.mIm.out());

    return aRes;

}

cImg::cImg(int aX,int aY) :
    mIm(aX,aY),
    mTIm(mIm),
    mSz(aX,aY)
{
}

cImg::cImg(const cImg &aIm) :
    mIm(aIm.mIm),
    mTIm(aIm.mTIm),
    mSz(aIm.mSz)
{
}


/************************************
 *
 * The starting point of the program
 *
 * *********************************/
int PPMD_Appariement_main(int argc,char** argv)
{
    std::cout << "TP appariement dense" << "\n";

    std::string aIm1Name, aIm2Name;
    int aSzW(2);
    int aPx(50);
	std::string aModeStr="SSD";

	// program's menu when called from terminal
    ElInitArgMain
    (
        argc,argv,
        LArgMain() << EAMC(aIm1Name,"First image name")
                   << EAMC(aIm2Name,"Second image name"),
        LArgMain() << EAM(aSzW,"Sz",true,"Window size")
                   << EAM(aPx,"Px",true,"Max Dispiarity") 
                   << EAM(aModeStr,"Mode",true,"Correlation mode (SSD, CORR, CENSUS)") 
    );
	eCorMode aMode = Str2Enum(aModeStr);
	std::cout << aMode << "\n";


	// Read the images

	// Do the matching

	// Save the result to images



    return EXIT_SUCCESS;
}
















