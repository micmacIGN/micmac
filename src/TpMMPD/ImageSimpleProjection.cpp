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
#include "TpPPMD.h"

/*
	script par JL, avril 2015
	Motivations: in the context of wildlife census by means of unmanned aerial surveys, the operator has to handle lot of aerial images
	These aerial images are not devoted to mapping (no sufficient overlap for Structure form Motion, by exemple)
	Operator are willing to measure the size of animals, a georectification of the images is thus required.
	 
   exemple :
	cd /media/jo/Data/Project_Photogrammetry/Exo_MM/Ex1_Felenne
       mm3d TestLib ImageRectification R00418.*.JPG Ori-sub-BL72/ Show=1 DeZoom=2
     
     to do: commenter, vérifier cohérence des résultats avec ortho, améliorer le Dezoom (plus efficient de le faire à la volée, et non pas apres le redressement).
*/

// List  of classes

// Class to store the application of Image Simple Rectification
class cISR_Appli;
// Contains the information about each image : Geometry (radiometric is handled by cISR_ColorImg
class cISR_Ima;
// for each images, 2 colorImg instance; one for the image itself, one for the Rectified image
class cISR_ColorImg;
// class for a color point
class cISR_Color;

// classes declaration

class cISR_Appli
{
    public :

        cISR_Appli(int argc, char** argv);
        const std::string & Dir() const {return mDir;}
        bool ShowArgs() const {return mShowArgs;}
        std::string NameIm2NameOri(const std::string &) const;
        cInterfChantierNameManipulateur * ICNM() const {return mICNM;}
		int mFlightAlti;
		int mDeZoom;
  
        void InitGeomTerrain();
     
    private :
        cISR_Appli(const cISR_Appli &); // To avoid unwanted copies

        void DoShowArgs1();
        void DoShowArgs2(int aKIm);
		
        std::string mFullName;
        std::string mDir;
        std::string mPat;
        std::string mOri;
        std::string mPrefixOut;
        std::list<std::string> mLFile;
        cInterfChantierNameManipulateur * mICNM;
        std::vector<cISR_Ima *>           mIms;
        bool                              mShowArgs;
        
};


class cISR_Ima
{
    public:
       cISR_Ima(cISR_Appli & anAppli,const std::string & aName,int aAlti,int aDZ, std::string & aPrefix);
    
       void InitMemImProj();
	   void InitGeom();
	   void GenTFW();
       void CalculImProj();
       Pt2di SzUV(){return mSzIm;}
       Pt2di SzXY(){return mSzImRect;}
       bool  DepthIsDefined(){return mCam->ProfIsDef();}
	   int 			   mAlti;
       int			   mZTerrain;
       int			   mDeZoom;
       int			   mBorder[4]; // the extent border of the rectified image
	   double		   mIGSD; // Initial ground sample distance 
	   double		   mFGSD; // Final ground sample distance, after resample (goal= decrease the size of the resulting rectified image)

    private :
       cISR_Appli &    mAppli;
       std::string     mName;
       std::string     mNameTiff;
	   Pt2di           mSzIm;	
	   Pt2di           mSzImRect;				
       std::string     mNameOri;
       std::string	   mPrefix;
       CamStenope *    mCam;
};

//Color image
class cISR_ColorImg
{
  public:
    cISR_ColorImg(std::string filename);
    cISR_ColorImg(Pt2di sz);
    ~cISR_ColorImg();
    cISR_Color get(Pt2di pt);
    cISR_Color getr(Pt2dr pt);
    void set(Pt2di pt, cISR_Color color);
    void write(std::string filename);
    cISR_ColorImg ResampleColorImg(double aFact);
    Pt2di sz(){return mImgSz;}
  protected:
    std::string mImgName;
    Pt2di mImgSz;
    Im2D<U_INT1,INT4> *mImgR;
    Im2D<U_INT1,INT4> *mImgG;
    Im2D<U_INT1,INT4> *mImgB;
    TIm2D<U_INT1,INT4> *mImgRT;
    TIm2D<U_INT1,INT4> *mImgGT;
    TIm2D<U_INT1,INT4> *mImgBT;
};

//color value class. just one point
class cISR_Color
{
  public:
    cISR_Color(U_INT1 r,U_INT1 g,U_INT1 b):mR(r),mG(g),mB(b){} // constructor
    void setR(U_INT1 r){mR=r;}
    void setG(U_INT1 g){mG=g;}
    void setB(U_INT1 b){mB=b;}
    U_INT1 r(){return mR;}
    U_INT1 g(){return mG;}
    U_INT1 b(){return mB;}
  protected:
    U_INT1 mR;
    U_INT1 mG;
    U_INT1 mB;
};

/********************************************************************/
/*                                                                  */
/*         cISR_ColorImg                                                 */
/*                                                                  */
/********************************************************************/


cISR_ColorImg::cISR_ColorImg(std::string filename) :
  mImgName(filename)
{
    Tiff_Im mTiffImg(mImgName.c_str());
    mImgSz.x=mTiffImg.sz().x;
    mImgSz.y=mTiffImg.sz().y;
    mImgR=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgG=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgB=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgRT=new TIm2D<U_INT1,INT4>(*mImgR);
    mImgGT=new TIm2D<U_INT1,INT4>(*mImgG);
    mImgBT=new TIm2D<U_INT1,INT4>(*mImgB);
    ELISE_COPY(mImgR->all_pts(),mTiffImg.in(),Virgule(mImgR->out(),mImgG->out(),mImgB->out()));
}


cISR_ColorImg::cISR_ColorImg(Pt2di sz) :
  mImgName(""),
  mImgSz(sz)
{
    mImgR=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgG=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgB=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgRT=new TIm2D<U_INT1,INT4>(*mImgR);
    mImgGT=new TIm2D<U_INT1,INT4>(*mImgG);
    mImgBT=new TIm2D<U_INT1,INT4>(*mImgB);
}

cISR_ColorImg::~cISR_ColorImg()
{
    delete mImgR;
    delete mImgG;
    delete mImgB;
    delete mImgRT;
    delete mImgGT;
    delete mImgBT;
} 

cISR_Color cISR_ColorImg::get(Pt2di pt) // the method get() return an objet "color" point
{
    return cISR_Color(mImgRT->get(pt,0),mImgGT->get(pt,0),mImgBT->get(pt,0));
}

cISR_Color cISR_ColorImg::getr(Pt2dr pt)
{
								// get (pt, 0) est plus robuste que get (pt), retourne 0 si le point est hors images
		return cISR_Color(mImgRT->getr(pt,0),mImgGT->getr(pt,0),mImgBT->getr(pt,0));
}

void cISR_ColorImg::set(Pt2di pt, cISR_Color color)
{
    U_INT1 ** aImRData=mImgR->data();
    U_INT1 ** aImGData=mImgG->data();
    U_INT1 ** aImBData=mImgB->data();
    aImRData[pt.y][pt.x]=color.r();
    aImGData[pt.y][pt.x]=color.g();
    aImBData[pt.y][pt.x]=color.b();
}

void cISR_ColorImg::write(std::string filename)
{
    ELISE_COPY
    (
        mImgR->all_pts(),
        Virgule( mImgR->in(), mImgG->in(), mImgB->in()) ,
        Tiff_Im(
            filename.c_str(),
            mImgSz,
            GenIm::u_int1,
            Tiff_Im::No_Compr,
            Tiff_Im::RGB,
            Tiff_Im::Empty_ARG ).out()
    );

}

cISR_ColorImg  cISR_ColorImg::ResampleColorImg(double aFact)
{
   Pt2di aSzR = round_up(Pt2dr(mImgSz)/aFact);

   cISR_ColorImg aResampled(aSzR);

   Fonc_Num aFInR = StdFoncChScale
                 (
                       this->mImgR->in_proj(),
                       Pt2dr(0,0),
                       Pt2dr(aFact,aFact)
                 );
  Fonc_Num aFInG = StdFoncChScale(this->mImgG->in_proj(),Pt2dr(0,0),Pt2dr(aFact,aFact));     
  Fonc_Num aFInB = StdFoncChScale(this->mImgB->in_proj(),Pt2dr(0,0),Pt2dr(aFact,aFact));          
   
    ELISE_COPY(aResampled.mImgR->all_pts(),aFInR,aResampled.mImgR->out());
    ELISE_COPY(aResampled.mImgG->all_pts(),aFInG,aResampled.mImgG->out());
    ELISE_COPY(aResampled.mImgB->all_pts(),aFInB,aResampled.mImgB->out());
   return aResampled;
}

/********************************************************************/
/*                                                                  */
/*         cISR_Ima                                                 */
/*                                                                  */
/********************************************************************/

// constructor of class ISR Image
cISR_Ima::cISR_Ima(cISR_Appli & anAppli,const std::string & aName,int aAlti,int aDZ, std::string & aPrefix) :
   mAlti   (aAlti),
   mZTerrain (0),
   mDeZoom (aDZ),
   mAppli  (anAppli),
   mName   	(aName), // Jpg image for e.g.
   mNameTiff (NameFileStd(mName,3,false,true,true,true)), // associated tif 3 channel image
   //mTifIm  (Tiff_Im::StdConvGen(mAppli.Dir() + mNameTiff,3,true)),
   mNameOri  (mAppli.NameIm2NameOri(mName)),
   mPrefix   (aPrefix),
   mCam      (CamOrientGenFromFile(mNameOri,mAppli.ICNM()))
{
   
}

void cISR_Ima::CalculImProj()
{
	//  For a given ground surface elevation, compute the rectified image (rectification==redressement)
	Pt2di aSz = this->SzXY();
	cISR_ColorImg ImCol(mNameTiff.c_str());
	cISR_ColorImg ImColRect(aSz);
    Pt2di aP;
	std::cout << "Beginning of rectification for oblique image " << this->mName  << "   -------------- \n";
	
	// Loop on every column and line of the rectified image
    for (aP.x=0 ; aP.x<aSz.x; aP.x++)
    {
		// compute X coordinate in ground/object geometry
		double aX=mBorder[0]+mIGSD * aP.x;
		
        for (aP.y=0 ; aP.y<aSz.y; aP.y++)
			{
			// compute Y coordinate in ground/object geometry
			double aY=mBorder[3]-mIGSD * aP.y;
			// define the point position in ground geometry
			Pt3dr aPTer(aX,aY,mZTerrain);
			// project this point in the initial image
			Pt2dr aPIm0 = mCam->R3toF2(aPTer); 
			//std::cout << " point 2D image  :: " << aPIm0 <<"    \n";
			//std::cout << " point 3D terrain :: " << aPTer <<"    \n";
			
			// get the radiometric value at this position
			cISR_Color aCol=ImCol.getr(aPIm0);
			// write the value on the rectified image
			ImColRect.set(aP,aCol);
			}
    }

	// write the rectified image in the working directory
	
	
    std::string aNameImProj= mPrefix+"-"+this->mName+".tif";
    
    // performed a resample of the rectified image.
    if (mDeZoom!=1)
		{
		cISR_ColorImg ImResampled = ImColRect.ResampleColorImg(mDeZoom);
		ImResampled.write(aNameImProj);
		std::cout << "Resampling of rectified image (dezoom factor of " << mDeZoom << ") \n";
		}
    else ImColRect.write(aNameImProj);
    
    // write the tfw file
    this->GenTFW();
    
    std::cout << "End of rectification for oblique image " << this->mName  << "  ------------------- \n \n"; 
}


void cISR_Ima::GenTFW()
           {
               std::string aNameTFW = mPrefix+"-"+this->mName + ".tfw";
               std::ofstream aFtfw(aNameTFW.c_str());
               aFtfw.precision(10);
               aFtfw << mFGSD << "\n" << 0 << "\n";
               aFtfw << 0 << "\n" << -mFGSD << "\n";
               aFtfw << mBorder[0] << "\n" << mBorder[3] << "\n";
               aFtfw.close();
           }

void cISR_Ima::InitMemImProj()
{
	// if the user has defined a Flight altitude, we assume the soil elevetion to be at Z=position of the camera-flight altitude.
	// else, the information of camera depth is used instead of flight altitude.
	if (mAlti==0) mAlti=static_cast<int>(mCam->GetProfondeur());
	// get the pseudo optical center of the camera (position XYZ of the optical center)
	Pt3dr OC=mCam->PseudoOpticalCenter();
	mZTerrain=static_cast<int>(OC.z-mAlti);
	// des fois l'info alti est notée dans mCam mais pas l'info Profondeur. c'est peut-etre uniquement le cas pour les mauvais orientation
	//if (mZTerrain<0) (mZTerrain=static_cast<int>(OC.z-mCam->GetAlti()));
	if (mZTerrain<0) {
			 std::cout << "For Image  " << this->mName  << " \n"; 
			 ELISE_ASSERT(false,"Ground Surface Elevation is below 0 (check FAlti)."); }
	// declare the 4 3Dpoints used for determining the XYZ coordinates of the 4 corners of the camera
	Pt3dr P1;
	Pt3dr P2;
	Pt3dr P3;
	Pt3dr P4;
	// project the 4 corners of the camera, ground surface assumed to be a plane
	mCam->CoinsProjZ(P1, P2, P3, P4, mZTerrain);
	// determine the ground sample distance.
	mIGSD=std::abs (mCam->ResolutionSol(Pt3dr(OC.x,OC.y,mZTerrain))); //initial ground sample distance
	mFGSD=mIGSD*mDeZoom; // final ground sample distance , different from Initial if dezoom is applied

	// determine  xmin,xmax,ymin, ymax
	double x[4]={P1.x,P2.x,P3.x,P4.x};
	double y[4]={P1.y,P2.y,P3.y,P4.y};
	double *maxx=std::max_element(x,x+4);
	double *minx=std::min_element(x,x+4);
	double *maxy=std::max_element(y,y+4);
	double *miny=std::min_element(y,y+4);
	//int border[4]={static_cast<int>(*minx),static_cast<int>(*maxx),static_cast<int>(*miny),static_cast<int>(*maxy)};
	mBorder[0]=static_cast<int>(*minx);
	mBorder[1]=static_cast<int>(*maxx);
	mBorder[2]=static_cast<int>(*miny);
	mBorder[3]=static_cast<int>(*maxy);
	// determine the size in pixel of the projected image - without dezoom
	int SzX=(mBorder[1]-mBorder[0])/mIGSD;
	int SzY=(mBorder[3]-mBorder[2])/mIGSD;
	
	mSzImRect = Pt2di(SzX,SzY);
}


/********************************************************************/
/*                                                                  */
/*         cISR_Appli                                               */
/*                                                                  */
/********************************************************************/


cISR_Appli::cISR_Appli(int argc, char** argv){
    // Reading parameter : check and  convert strings to low level objects
    mShowArgs=false;
    int mFlightAlti = 0;
    int mDeZoom=1;
    std::string mPrefixOut="Rectified";
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mFullName,"Full Name (Dir+Pat)")
                    << EAMC(mOri,"Used orientation, must be a projected coordinate system (no WGS, relative or RTL orientation)"),
        LArgMain()  << EAM(mPrefixOut,"PrefixOut",true,"Prefix for the name of the resulting rectified image (ex 'toto' --> toto-R000567.JPG.tif), by default == 'Rectified'")
                    << EAM(mFlightAlti,"FAlti",true,"The flight altitude Above Ground Level. By default, use the flight alti computed by aerotriangulation")
                    << EAM(mDeZoom,"DeZoom",true,"DeZoom of the original image, by default no dezoom")
                    << EAM(mShowArgs,"Show",true,"Print details during the processing")
    );
    // Initialize name manipulator & files
    SplitDirAndFile(mDir,mPat,mFullName);
    // define the "working directory" of this session
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    // create the list of images starting from the regular expression (Pattern)
    mLFile = mICNM->StdGetListOfFile(mPat);

    StdCorrecNameOrient(mOri,mDir);

	// the optional argument Show = True, print the number of images as well as the names of every images
    if (mShowArgs) DoShowArgs1();
    
    // Initialize the images list in the class cISR_Ima
    for (
              std::list<std::string>::iterator itS=mLFile.begin();
              itS!=mLFile.end();
              itS++
              )
     {
           cISR_Ima * aNewIm = new  cISR_Ima(*this,*itS,mFlightAlti,mDeZoom,mPrefixOut);
           mIms.push_back(aNewIm);   
           
           //test if there are enough information about flight altitude (either given by the aerotriangulation or the user with FAlti)
           
           if (mFlightAlti==0 && (aNewIm->DepthIsDefined()==0)) ELISE_ASSERT(false,"Flight Altitude not known (use FAlti)");
     }
	
	// Define the ground footprint (image swath) of every rectified images
	InitGeomTerrain();
	
	// Compute all rectified images
         for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
         {
            // print details if asked:
			if (mShowArgs) DoShowArgs2(aKIm);
             mIms[aKIm]->CalculImProj();
		  }
}

void cISR_Appli::InitGeomTerrain()
{
    for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
    {
        cISR_Ima * anIm = mIms[aKIm];
		// Define the ground footprint of each georectified images
        anIm->InitMemImProj() ;
    }
}

std::string cISR_Appli::NameIm2NameOri(const std::string & aNameIm) const
{
    return mICNM->Assoc1To1
    (
        "NKS-Assoc-Im2Orient@-"+mOri+"@",
        aNameIm,
        true
    );
}

void cISR_Appli::DoShowArgs1()
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

void cISR_Appli::DoShowArgs2(int aKIm)
{
				cISR_Ima * anIm = mIms[aKIm];
   				std::cout << "Flight altitude [m]: 	" << anIm->mAlti << "  \n";
				std::cout << "Altitude of the gound surface  : 	" << anIm->mZTerrain << " \n";
				std::cout << "Initial Ground Sample Distance :	" << anIm->mIGSD << " \n";
				std::cout << "Ground Sample Distance of Rectified images : " << anIm->mFGSD << " \n";
				std::cout << "Rectified image size in pixel (before Resample) : " <<  anIm->SzXY() <<"  \n";		
				std::cout << "Rectified image X coverage [m] : 	" <<  anIm->mBorder[1]-anIm->mBorder[0] << "  \n";
}

int ImageRectification(int argc,char ** argv)
{
     cISR_Appli anAppli(argc,argv);
	
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
