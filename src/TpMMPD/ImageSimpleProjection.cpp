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
	Operator are willing to measure the size of animals, a rough georeferencing of the images is thus required.
	 
   Par exemple :
	cd /media/jo/Data/Project_Photogrammetry/Exo_MM/Ex1_Felenne
       mm3d TestLib ImageProjection R00418.*.JPG Ori-sub-BL72/ Show=1 DeZoom=2
     
     to do: commenter le code et restructurer, export en RGB, vérifier cohérence des résultats avec ortho, améliorer le Dezoom (pr le moment, c'est un "echantillonnage" de l'image initiale).
*/

// List  of classes

// Class to store the application of image simple projection
class cISP_Appli;
// Contains the information to store each image : Radiometry & Geometry
class cISP_Ima;

// classes declaration

class cISP_Ima
{
    public:
       cISP_Ima(cISP_Appli & anAppli,const std::string & aName,int aAlti,bool aDetail,int aDZ);
 
       //Video_Win *  W() {return mW;}
       //Pt2dr ClikIn();
       
       void InitMemImProj();
       
	   void InitGeom();
		
	   void GenTFW();
       void CalculImProj();
       
       Pt2di Sz(){return mSz;}

    private :
       cISP_Appli &    mAppli;
       std::string     mName;
       Tiff_Im         mTifIm;
       Pt2di           mSz;
       Im2D_U_INT1     mIm;
       Im2D_U_INT1     mImProj;
	   int			   mBorder[4];
	   double		   mGSD;
       //Video_Win *     mW;
       std::string     mNameOri;
       CamStenope *    mCam;
       int 			   mAlti;
       int			   mZTerrain;
       bool			   mDetail;
       int			   mDeZoom;

};


class cISP_Appli
{
    public :

        cISP_Appli(int argc, char** argv);
        const std::string & Dir() const {return mDir;}
        bool ShowArgs() const {return mShowArgs;}
        std::string NameIm2NameOri(const std::string &) const;
        cInterfChantierNameManipulateur * ICNM() const {return mICNM;}
		int mFlightAlti;
		int mDeZoom;
  
        void InitGeomTerrain();
     
    private :
        cISP_Appli(const cISP_Appli &); // To avoid unwanted copies

        void DoShowArgs();
		
        std::string mFullName;
        std::string mDir;
        std::string mPat;
        std::string mOri;
        std::list<std::string> mLFile;
        cInterfChantierNameManipulateur * mICNM;
        std::vector<cISP_Ima *>          mIms;
        bool                              mShowArgs;
        
};

/********************************************************************/
/*                                                                  */
/*         cISP_Ima                                                 */
/*                                                                  */
/********************************************************************/

// constructor of class ISP Image
cISP_Ima::cISP_Ima(cISP_Appli & anAppli,const std::string & aName,int aAlti,bool aDetail,int aDZ) :
   mAppli  (anAppli),
   mName   (aName),
   mTifIm  (Tiff_Im::StdConvGen(mAppli.Dir() + mName,1,true)),
   mSz     (mTifIm.sz()),
   mIm     (mSz.x,mSz.y),
   mImProj (1,1), // Do not know the size for now
   //mBorder({0,0,0,0}),
   //mW      (0),
   mNameOri  (mAppli.NameIm2NameOri(mName)),
   mCam      (CamOrientGenFromFile(mNameOri,mAppli.ICNM())),
   mAlti   (aAlti),
   mZTerrain (0),
   mDetail (aDetail),
   mDeZoom (aDZ)
{
   ELISE_COPY(mIm.all_pts(),mTifIm.in(),mIm.out());
}


void cISP_Ima::CalculImProj()
{
	// For a given ground surface elevation, compute the rectified image
    TIm2D<U_INT1,INT> aTIm(mIm);  // Initial image
    TIm2D<U_INT1,INT> aTImProj(mImProj); // image projected
	Pt2di aSz = aTImProj.sz();
    Pt2di aP;
	std::cout << " Begin projection for image " << this->mName  << "   -------------- \n";
	
	// Loop on every column and line of the projected image
    for (aP.x=0 ; aP.x<aSz.x; aP.x++)
    {
		// compute X coordinate in ground/object geometry
		double aX=mBorder[0]+mGSD * aP.x;
		
        for (aP.y=0 ; aP.y<aSz.y; aP.y++)
			{
			// compute Y coordinate in ground/object geometry
			double aY=mBorder[3]-mGSD * aP.y;
			// define the point position in ground geometry
			Pt3dr aPTer(aX,aY,mZTerrain);
			// project this point in the initial image
			Pt2dr aPIm0 = mCam->R3toF2(aPTer); 
			// get the radiometric value at this position
			float aVal = aTIm.getr(aPIm0,0);
			// wirte the value on the projected image
			aTImProj.oset(aP,round_ni(aVal));
			//std::cout << " point 2D image  :: " << aPIm0 <<"    \n";
			//std::cout << " point 3D terrain :: " << aPTer <<"    \n";
			//std::cout << "Pixels Value : " << aVal << "  \n";	
			}
    }

	// write the projected image in the working directory
    std::string aNameImProj=this->mName+".Projected.tif";
    
    // initialize a Tiff_Im object
    Tiff_Im  aTF
         (
            aNameImProj.c_str(),
            mImProj.sz(),
            GenIm::real4,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero
                 //Tiff_Im::RGB
           );
    // write the file
	ELISE_COPY(mImProj.all_pts(),mImProj.in(),aTF.out());
    // create the tfw file for georeferencing the image  
    GenTFW();
    
    std::cout << " End projection for image " << this->mName  << "  -------------------  \n"; 

}


void cISP_Ima::GenTFW()
           {
               std::string aNameTFW = this->mName + ".Projected.tfw";
               std::ofstream aFtfw(aNameTFW.c_str());
               aFtfw.precision(10);
               aFtfw << mGSD << "\n" << 0 << "\n";
               aFtfw << 0 << "\n" << -mGSD << "\n";
               aFtfw << mBorder[0] << "\n" << mBorder[3] << "\n";
               aFtfw.close();
           }


void cISP_Ima::InitMemImProj()
{
	// if the user has defined a Flight altitude, we assume the soil elevetion to be at Z=position of the camera-flight altitude.
	// else, the information of camera depth is used instead of flight altitude.
	if (mAlti==0) mAlti=static_cast<int>(mCam->GetProfondeur());
	// get the pseudo optical center of the camera (position XYZ of the optical center)
	Pt3dr OC=mCam->PseudoOpticalCenter();
	mZTerrain=static_cast<int>(OC.z-mAlti);
	if (mZTerrain<0) ELISE_ASSERT(false,"Ground Surface Elevation is below 0."); 
	// declare the 4 3Dpoints used for determining the XYZ coordinates of the 4 corners of the camera
	Pt3dr P1;
	Pt3dr P2;
	Pt3dr P3;
	Pt3dr P4;
	// project the 4 corners of the camera, ground surface assumed to be a plane
	mCam->CoinsProjZ(P1, P2, P3, P4, mZTerrain);
	// determine the ground sample distance.
	double aGSD=std::abs (mCam->ResolutionSol(Pt3dr(OC.x,OC.y,mZTerrain)));
	mGSD=aGSD*mDeZoom;

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
	// determine the size in pixel of the projected image
	int SzX=(mBorder[1]-mBorder[0])/mGSD;
	int SzY=(mBorder[3]-mBorder[2])/mGSD;
	
	if (mDetail) 
		{
			std::cout << "Flight altitude [m]: 	" <<  mAlti << "  \n";
			std::cout << "Altitude of the gound surface  : 	" << mZTerrain << " \n";
			std::cout << "Initial Ground Sample Distance :	" << aGSD << " \n";
			std::cout << "Ground Sample Distance of projected images : " << mGSD << " \n";
			std::cout << "Projected image size : SzX: " <<  SzX << ", SzY " << SzY << "  \n";		
			std::cout << "Projected image X coverage [m] : 	" <<  mBorder[1]-mBorder[0] << "  \n";
		}
	// resize the projected image
    mImProj.Resize(Pt2di(SzX,SzY));

}


/********************************************************************/
/*                                                                  */
/*         cISP_Appli                                               */
/*                                                                  */
/********************************************************************/


cISP_Appli::cISP_Appli(int argc, char** argv){
    // Reading parameter : check and  convert strings to low level objects
    mShowArgs=false;
    int mFlightAlti = 0;
    int mDeZoom=1;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mFullName,"Full Name (Dir+Pat)")
                    << EAMC(mOri,"Used orientation, must be a projected coordinate system (no WGS, relative or RTL orientation)"),
        LArgMain()  << EAM(mShowArgs,"Show",true,"Give details on args")
                    << EAM(mFlightAlti,"FAlti",true,"The flight altitude Above Ground Level. By default, use the flight alti computed by aerotriangulation")
                    << EAM(mDeZoom,"DeZoom",true,"DeZoom of the original image, by default no dezoom")
    );
    // Initialize name manipulator & files
    SplitDirAndFile(mDir,mPat,mFullName);
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    // create the list of images starting from the regular expression (Pattern)
    mLFile = mICNM->StdGetListOfFile(mPat);

    StdCorrecNameOrient(mOri,mDir);

	// the optional argument Show = True, print the number of images as well as the names of every images
    if (mShowArgs) DoShowArgs();
	
    // Initialize the images list in the class cISP_Ima
    for (
              std::list<std::string>::iterator itS=mLFile.begin();
              itS!=mLFile.end();
              itS++
              )
     {
           cISP_Ima * aNewIm = new  cISP_Ima(*this,*itS,mFlightAlti,mShowArgs,mDeZoom);
           mIms.push_back(aNewIm);   
     }
	
	// Define the ground footprint of every projected images
	InitGeomTerrain();
	
	// Compute all projected images
         for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
              mIms[aKIm]->CalculImProj();

}

void cISP_Appli::InitGeomTerrain()
{
    for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
    {
        cISP_Ima * anIm = mIms[aKIm];
		// Define the ground footprint of every projected images
        anIm->InitMemImProj() ;
    }
}


std::string cISP_Appli::NameIm2NameOri(const std::string & aNameIm) const
{
    return mICNM->Assoc1To1
    (
        "NKS-Assoc-Im2Orient@-"+mOri+"@",
        aNameIm,
        true
    );
}

void cISP_Appli::DoShowArgs()
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

int ImageProjection(int argc,char ** argv)
{
     cISP_Appli anAppli(argc,argv);

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
