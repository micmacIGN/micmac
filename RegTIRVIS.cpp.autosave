#include "StdAfx.h"
#include <fstream>
#include "Image.h"
#include "msd.h"
#include "Keypoint.h"
#include "lab_header.h"
#include "../../uti_image/Digeo/Digeo.h"
#include "DescriptorExtractor.h"
#include "Arbre.h"

#include "../../uti_image/Digeo/DigeoPoint.h"

#define distMax 0.75
#define rayMax 0.5

/*******************************************************/

//Objective:
// Registration problem


/*1. Extract extern orientation from the computed orientation files for Visual Images
 2. Apply the known homography to thermal images: For each tir image, there is an
   homologous visual image known by name

   N.B: A criterion to evaluate the best homography transformation based on another data set is to be defined

 3. Construct a thermal orientation directory that contains all ori files for thermal images
 4. Apply the known homography predictor to compute therm-optical correspondences
 5. Use command Bar already available in MicMac in order to compute a robust 3D similarity between thermal and optical sets
    */

/*******************************************************/

struct PtAndIdx
{
    Pt2dr Pt;
    int CaptureIndex;
};

/*******************************************************/

struct ImAndPt
{
    string ImageName;
    Pt2dr MesureImage;
};

/*******************************************************/
// Dislay a vector

template <class tData>
void DisplayVector(std::vector<tData> V)
{
    for (int i=0;i<V.size();i++)
    {
        std::cout<<V.at(i)<<" ";
    }
    std::cout<<"\n";
}


/*********************************************************/
//Check if and index is within range

bool WithinLimits(uint index, vector<KeyPoint> Kps)
{
    if (index>0 && index<Kps.size())
    {
        return true;
    }
    else
    {
        return false;
    }
}
/*********************************************************/
//Check for inclusion in depth map
bool IncludedInDepthIm(Pt2dr Point, Pt2di DepthSize)
{
    if (Point.x>0 && Point.x<(DepthSize.x-1) && Point.y>0 && Point.y< (DepthSize.y-1))
    {
        return true;
    }
    else
    {
        return false;
    }
}
//===============================================================================//
// Resize function: resize images using an interpolation scheme: Bilinear here
//===============================================================================//
template <class Type, class TyBase>
void Resizeim(Im2D<Type,TyBase> & im, Im2D<Type,TyBase> & Out, Pt2dr Newsize)
{
    Out.Resize(Pt2di(Newsize));
    float tx=(im.sz().x-1)/Newsize.x;
    float ty=(im.sz().y-1)/Newsize.y;


    for (int i=0;i<Newsize.x;i++)
    {
        for(int j=0;j<Newsize.y;j++)
        {
            Pt2dr PP(tx*i,ty*j);
            Pt2di Dst(i,j);
            REAL8 RetVal;
            RetVal=im.BilinIm()->Get(PP);
            if (im.TypeEl()==GenIm::u_int1|| im.TypeEl()==GenIm::u_int2)
            {
                if(RetVal<im.vmin() || RetVal>im.vmax())
                {
                    RetVal=im.vmax();
                }

                Out.SetI(Dst,round(RetVal));
            }
            else
            {
                Out.SetR(Dst,RetVal);
            }
        }
    }
}
//===============================================================================//
// Estimate Homograhy via multiple iterations on keypoints which have not been matched during
// the correspondence search phase
//===============================================================================//

void EnrichKps(std::vector< KeyPoint > Kpsfrom, ArbreKD * Tree, cElHomographie &Homog, int NbIter)
{
    ElSTDNS set< pair<int,Pt2dr> > Voisins ; // where to put nearest neighbours

    for (int i=0; i<NbIter; i++)
    {
        ElPackHomologue HomologousPts;  // THE good keypoints don't need to be erased they need to be enriched RANSAC should provide a Mask
        // of chosen couples
        //HomologousPts.Cple_Add(ElCplePtsHomologues(P1,P1,1.0);
        std::vector<KeyPoint>::iterator kp=Kpsfrom.begin();
        for(;kp!=Kpsfrom.end();kp++)
        {
            Pt2dr aPoint(kp->getPoint().x,kp->getPoint().y);
            Pt2dr aPointOut=Homog.Direct(aPoint);
            Voisins.clear();
            Tree->voisins(aPointOut,distMax, Voisins);
            if (Voisins.size()>0)
            {
                Pt2dr Ptfound(Voisins.begin()->second);
                HomologousPts.Cple_Add(ElCplePtsHomologues(aPoint, Ptfound));
            }
        }
        double Ecart, Quality;
        bool Ok;
        Homog=Homog.RobustInit(Ecart,&Quality,HomologousPts,Ok,50,80.0,2000);

        std::cout<<" Computed Homography at  "<<i<<  "  Iteration\n";
        Homog.Show();
        std::cout<<" Quality parameters \n";
        std::cout<<"========= >Ecart "<<Ecart<<endl;
        std::cout<<"========= >Quality "<<Quality<<endl;
        std::cout<<"========= >If Ok=1: "<<Ok<<endl;
    }
    //delete mImgIdx;
}

//===============================================================================//
// Call for wallis filter: locally equalized image
//===============================================================================//

void wallis( Im2D<U_INT1,INT> &image, Im2D<U_INT1,INT> &WallEqIm)
{

    // image should be changed to gray space

    cout <<"==============> Image Filtering via Wallis algorithm....."<< endl;

    int n = image.sz().x;
    int m = image.sz().y;

    std::cout<<"Image size "<<n<<"   "<<m<<endl;

    // Block dimension in i and j
    int dim_n = 40, dim_m = 40;

    int N_Block = n/dim_n;
    cout<<"N_Block\t"<<N_Block<<endl;
    int M_Block = m/dim_m;
    cout<<"M_Block\t"<<M_Block<<endl;
    int resto_n = n%dim_n;
    int resto_m = m%dim_m;

    int dimension_x[N_Block];
    int dimension_y[M_Block];

    dim_n = dim_n + resto_n/N_Block;
    dim_m = dim_m + resto_m/M_Block;
    resto_n = n%dim_n;
    resto_m = m%dim_m;

    int i;
    for (i=0; i < N_Block; i++)
    {
        if (resto_n>0)
        {
        dimension_x[i] = dim_n+1;
        resto_n--;
        }
        else
        {
        dimension_x[i] = dim_n;
        }
    }


    for (i=0; i < M_Block; i++)
    {
        if (resto_m>0)
        {
        dimension_y[i] = dim_m+1;
        resto_m--;
        }
        else
        {
        dimension_y[i] = dim_m;
        }
        //printf("%d\n", dimension_y[i]);
    }

    // c is the CONTRAST expansion constant [0.7-1.0]
    // to reduce the enhancement of noise c should be reduced
    // it has a much stronger influence on noise than sf
    // lower values produce very little contrast and detail
    // values closer to 1 produce a highly contrasted image with greater detail
    double c = 0.8;

    // sf is the target value of the LOCAL STANDARD DEVIATION in a i,j window [50.0-80.0]
    // the value of sf should decrease with decreasing window dimensions(i,j) to avoid increase of noise
    // it decides the contrast of image; higher values result in a greater contrast stretch
    // producing higher local contrast and greater detail
    double sf =  80.0;

    // b is the BRIGHTNESS forcing constant [0.5-1.0]
    // to keep primary image gray mean b has to be small
    // 0 will keep the original pixel values
    // 1 will generate an output image equal to the wallis filter specified
    double b = 0.9;

    // mf is the target value of the LOCAL MEAN in a i,j window [127.0-140.0]
    // an higher value wil brighten the image
    double mf = 127.0;

    int px = 0, py = 0;

    Im2D<REAL4,REAL8> Coeff_R0 = Im2D<REAL4,REAL8>(N_Block, M_Block);
    Im2D<REAL4,REAL8> Coeff_R1 = Im2D<REAL4,REAL8>(N_Block, M_Block);
    cout <<"Coeff_R0 "<<	Coeff_R0.sz() <<endl;
    cout <<"Coeff_R1"<<	Coeff_R1.sz() <<endl;
    // computing mean and standard deviation in every (dim_n*dim_m) window

    //
    Symb_FNum aTF=Rconv(image.in());


    for(int i=0; i<N_Block; i++)
    {
        py = 0;
        for(int j=0; j<M_Block; j++)
        {
            Pt2di aP0(px,py);
            Pt2di aSz(dimension_x[i],dimension_y[j]);
            /*********************************************************/
            //code from CCP_statimage.cpp
            double aSP,aSomZ,aSomZ2,aZMin,aZMax;

            ELISE_COPY
            (
                rectangle(aP0,aP0+aSz),
                Virgule(1,aTF,Square(aTF)),
                Virgule
                (
                     sigma(aSP),
                     sigma(aSomZ)|VMax(aZMax)|VMin(aZMin),
                     sigma(aSomZ2)
                )
            );

            aSomZ /= aSP;
            aSomZ2 /= aSP;
            aSomZ2 -= ElSquare(aSomZ);

            double StdDev =sqrt(ElMax(0.0,aSomZ2));
            /*************************************************************/
            py += dimension_y[j];

            double r1 = c*sf/(c*StdDev + (1-c)*sf);				        //Baltsavias
            //double r1 = c*stDev.val[0]/(c*stDev.val[0] + (sf/c));		//Fraser
            //double r1 = c*sf/(c*stDev.val[0] + (sf/c));				//Xiao
            double r0 = b*mf + (1 - b - r1)*aSomZ;
            //std::cout<<"mean "<<aSomZ<<"stdev "<<StdDev<<endl;
          //  std::cout<<"r0 " <<r0<< "    "<<" r1 "<<r1<<endl;
            Coeff_R1.SetR(Pt2di(i,j),r1);
            Coeff_R0.SetR(Pt2di(i,j),r0);
        }
        px += dimension_x[i];
    }

    Im2D<REAL4,REAL8> Coeff_R00 = Im2D<REAL4,REAL8>(image.sz().x, image.sz().y);
    Im2D<REAL4,REAL8> Coeff_R11 = Im2D<REAL4,REAL8>(image.sz().x, image.sz().y);

    Resizeim<REAL4,REAL8>(Coeff_R1, Coeff_R11, Pt2dr(image.sz()));
    Resizeim<REAL4,REAL8>(Coeff_R0, Coeff_R00, Pt2dr(image.sz()));



    Im2D<REAL4,REAL8> Imagefloat=Im2D<REAL4,REAL8>(image.sz().x,image.sz().y);

   // Multiply term by term Imagefloat and Coeff_R11 and add Coeff_R00 terms
    U_INT1 ** dtim0=image.data();
    REAL4 ** dta=Imagefloat.data();
    REAL4 ** dtcoeff1=Coeff_R11.data();
    REAL4 ** dtcoeff0=Coeff_R00.data();

    float Minvalue=std::numeric_limits<float>::max();
    float Maxvalue=std::numeric_limits<float>::min();

    for (int x=0;x<Imagefloat.sz().x;x++)
    {
        for(int y=0;y<Imagefloat.sz().y;y++)
        {
           dta[y][x]=(float)dtim0[y][x];
           dta[y][x]*=dtcoeff1[y][x];
           dta[y][x]+=dtcoeff0[y][x];
           Maxvalue= (Maxvalue <= dta[y][x]) ? dta[y][x]: Maxvalue;
           Minvalue= (Minvalue >= dta[y][x]) ? dta[y][x]: Minvalue;
        }
    }
    ELISE_COPY(
                WallEqIm.all_pts(),
                (Imagefloat.in()-Minvalue)*255.0/(Maxvalue-Minvalue),
                WallEqIm.out()
               );
}
//===============================================================================//
/*      Routine that processes an image: RGB--> Lab and Wallis filter            */
//===============================================================================//
void Migrate2Lab2wallis(Tiff_Im &image, Im2D<U_INT1,INT> & Output)
{
    RGB2Lab_b rgbtolab=RGB2Lab_b(3,0,0,0,true);
    Pt2di mImgSz(image.sz().x,image.sz().y);

    Im2D<U_INT1,INT>* mImgR(0);
    Im2D<U_INT1,INT>* mImgG(0);
    Im2D<U_INT1,INT>* mImgB(0);

    if (image.nb_chan()==3)
    {
        mImgR=new Im2D<U_INT1,INT>(mImgSz.x,mImgSz.y);
        mImgG=new Im2D<U_INT1,INT>(mImgSz.x,mImgSz.y);
        mImgB=new Im2D<U_INT1,INT>(mImgSz.x,mImgSz.y);
        ELISE_COPY(mImgR->all_pts(),image.in(),Virgule(mImgR->out(),mImgG->out(),mImgB->out()));
    }
    else
    {
        mImgR=new Im2D<U_INT1,INT>(mImgSz.x,mImgSz.y);
        mImgG=new Im2D<U_INT1,INT>(mImgSz.x,mImgSz.y);
        mImgB=new Im2D<U_INT1,INT>(mImgSz.x,mImgSz.y);
        ELISE_COPY(mImgR->all_pts(),image.in(),mImgR->out());
        ELISE_COPY(mImgG->all_pts(),image.in(),mImgG->out());
        ELISE_COPY(mImgB->all_pts(),image.in(),mImgB->out());
    }

    //specify out images

    Im2D<U_INT1,INT>* mImgRo=new Im2D<U_INT1,INT>(mImgSz.x,mImgSz.y);
    Im2D<U_INT1,INT>* mImgGo=new Im2D<U_INT1,INT>(mImgSz.x,mImgSz.y);
    Im2D<U_INT1,INT>* mImgBo=new Im2D<U_INT1,INT>(mImgSz.x,mImgSz.y);


    rgbtolab.operator ()(mImgR,mImgG,mImgB,mImgRo,mImgGo,mImgBo);

    // Convert image to gray

   Im2D<U_INT1,INT> IMAGE_GR(image.sz().x,image.sz().y);
   ELISE_COPY(
               IMAGE_GR.all_pts(),
               (mImgRo->in()+mImgGo->in()+mImgBo->in())/3.0, // or take only the lightness component
                IMAGE_GR.out()
               );


   delete mImgB; delete mImgBo;
   delete mImgG; delete mImgGo;
   delete mImgR; delete mImgRo;

// apply wallis filter

   Output.Resize(IMAGE_GR.sz());
   wallis(IMAGE_GR,Output);
}

/*===============================================================================*/
/*               Write Keypoints to file : No need for recomputing them         */
/*===============================================================================*/
 void StoreKps(std::vector<KeyPoint> KPs, string file)
 {

     FILE * aFOut = FopenNN(file.c_str(),"w","out");
     std::vector<KeyPoint>::iterator kp=KPs.begin();
     while(kp!=KPs.end())
     {
         fprintf(aFOut,"%f %f %f %f\n",kp->getPoint().x,kp->getPoint().y,kp->getSize(),kp->getAngle());
         kp++;
     }
     ElFclose(aFOut);
     //delete aFOut;
 }

/*===============================================================================*/
/*               Define a method to apply an homography to image                 */
/*===============================================================================*/
void ApplyHomography(std::string fileim1, std::string fileim2 , cElHomographie H, std::string Outputfilename)
{
    ColorImg ImageDeCouleur(fileim1);
    ColorImg Imagevis(fileim2);
    //  generat the rectified image by using an homography, faster than with R3toF2
    Pt2di aSz =ImageDeCouleur.sz();
    ColorImg ImColRect(Imagevis.sz());
    Pt2di aP;
    std::cout << "Beginning of rectification by homography for oblique image " <<endl;

    // Loop on every column and line of the rectified image
    for (aP.x=0 ; aP.x<aSz.x; aP.x++)
    {
        // compute X coordinate in ground/object geometry
        double aX=aP.x;

        for (aP.y=0 ; aP.y<aSz.y; aP.y++)
            {
            double aY=aP.y;
            Pt2dr aPoint(aX,aY);
            // project this point in the initial image using the homography relationship
            //cElHomographie Inv=H.Inverse();
            Pt2dr aPIm0=H.Direct(aPoint);
            Pt2di aPImr((INT)aPIm0.x, (INT) aPIm0.y);

            // get the radiometric value at this position
            Color aCol=ImageDeCouleur.getr(aPoint);
            // write the value on the rectified image
            ImColRect.set(aPImr,aCol);
            }
    }
    ImColRect.write(Outputfilename.c_str());
    std::cout << "End of rectification by homography for oblique image " <<"\n";
}

/*===============================================================================*/
/*               apply homography to a set of points  KeyPoint                   */
/*===============================================================================*/

std::vector<Pt2dr> NewSetKpAfterHomog(std::vector<KeyPoint> Kps, cElHomographie H)
{
    std::vector<Pt2dr> Newkps;
    for (uint i=0;i<Kps.size();i++)
    {

        Pt2dr Pt(Kps.at(i).getPoint().x,Kps.at(i).getPoint().y);
        Pt2dr Pth=H.Direct(Pt);
        Newkps.push_back(Pth);
    }
    return Newkps;
}

/*===============================================================================*/
/*               apply homography to a set of points SiftPoint                   */
/*===============================================================================*/

std::vector<Pt2dr> NewSetKpAfterHomog(vector<SiftPoint> Kps, cElHomographie H)
{
    std::vector<Pt2dr> Newkps;
    std::vector<SiftPoint>::iterator Kp=Kps.begin();
    for (;Kp!=Kps.end();Kp++)
    {

        Pt2dr Pt(Kp->x,Kp->y);
        Pt2dr Pth=H.Direct(Pt);
        Newkps.push_back(Pth);
    }
    return Newkps;
}

/*===============================================================================*/
/*              Move from SiftPoint to KeyPoint                                  */
/*===============================================================================*/

std::vector<KeyPoint> FromSiftP2KeyP(std::vector<SiftPoint> Kps)
{
    std::vector<KeyPoint> Newkps;
    std::vector<SiftPoint>::iterator Kp=Kps.begin();
    for (;Kp!=Kps.end();Kp++)
    {

        KeyPoint Pt(Kp->x,Kp->y,Kp->scale,Kp->angle,0.0);
        Newkps.push_back(Pt);
    }
    return Newkps;
}
//===============================================================================//
// Checks if file is already present in folder
bool DoesFileExist( const char * FileName )
{
    #if (ELISE_unix)
        FILE* fp = NULL;
        fp = fopen( FileName, "rb" );
        if( fp != NULL )
        {
            fclose( fp );
            return true;
        }
    #endif
    return false;
}


//===============================================================================//
/*                          Read keypoints from file                             */
/*                          Format:x y scale orientation                         */
//===============================================================================//

void Readkeypoints(std::vector<KeyPoint> &Kps, string file)
{
    ELISE_fp aFIn(file.c_str(),ELISE_fp::READ);


    char * aLine;
    while ((aLine = aFIn.std_fgets()))
    {
        Pt2df aPIm;
        float scale=0.0;
        float angle=0.0;
        int aNb = sscanf(aLine,"%f %f %f %f",&aPIm.x,&aPIm.y,&scale,&angle);
        //std::cout<<"File has been read\n";
        //std::cout<<aNb<<"\n";
        ELISE_ASSERT(aNb==4,"Could not read: Format:x y scale orientation ");
        KeyPoint KP(aPIm,scale,angle,0.0); // Orientations given by the MSD detector are ambiguous (angle =0.0)
        Kps.push_back(KP);
    }
}
//===============================================================================//
/*                           Get Thermal for RGB image name                      */
//===============================================================================//
string WhichThermalImage(string VisualIm,std::vector< string > ThermalSet)
{
    cElRegex rgx("VIS(.*).tif",10);  // need to change according to pattern
    std::string aNameMatch;
    if (rgx.Match(VisualIm))
    {
        aNameMatch=rgx.KIemeExprPar(1);
    }
    string Imagetoreturn;
    std::size_t foundCommonPattern;
    for (uint i=0;i<ThermalSet.size();i++)
    {
        foundCommonPattern=ThermalSet.at(i).find(aNameMatch);
        if (foundCommonPattern!=std::string::npos)
        {
            Imagetoreturn=ThermalSet.at(i);
        }
        /*else
        {
            std::cout<<"Homologous thermal Image having the saem pattern is nott found\n";
        }*/

    }
    return Imagetoreturn;

}
//===============================================================================//
/*                           Get RGB for Thermal image name                      */
//===============================================================================//
string WhichVisualImage(string tirIm,std::vector< string > VisualSet)
{
    cElRegex rgx("TIR(.*).tif",10);
    std::string aNameMatch;
    if (rgx.Match(tirIm))
    {
        aNameMatch=rgx.KIemeExprPar(1);
    }
    string Imagetoreturn;
    std::size_t foundCommonPattern;
    for (uint i=0;i<VisualSet.size();i++)
    {
        foundCommonPattern=VisualSet.at(i).find(aNameMatch);
        if (foundCommonPattern!=std::string::npos)
        {
            Imagetoreturn=VisualSet.at(i);
        }
        /*else
        {
            std::cout<<"Homologous thermal Image having the saem pattern is nott found\n";
        }*/

    }
    return Imagetoreturn;

}
//===============================================================================//
/*       Parse a file GrapheHomol to get images in scope of one image            */
//===============================================================================//
void ParseHomol(string MasterImage, std::vector< cCpleString> ImCpls,std::vector< string > &ListHomol)
{
    ListHomol.clear();
    for (uint i=0;i<ImCpls.size();i++)
    {
        if (MasterImage.compare(ImCpls.at(i).N1())==0)
        {
            ListHomol.push_back(ImCpls.at(i).N2());
        }
    }
}

//===============================================================================//
/*                           OrientedImage class                                 */
//===============================================================================//
class Orient_Image
{
 public:
 Orient_Image
 (
  std::string aOriIn,
  std::string aName,
  cInterfChantierNameManipulateur * aICNM
 );
 std::string getName(){return mName;}
 CamStenope * getCam(){return mCam;}
 std::string getOrifileName(){return mOriFileName;}

 protected:
 CamStenope   * mCam;
 std::string  mName;
 std::string mOriFileName;
 };

Orient_Image::Orient_Image
 ( std::string aOriIn,
 std::string aName,
 cInterfChantierNameManipulateur * aICNM):
 mName(aName),mOriFileName(aOriIn+"Orientation-"+mName+".xml")
{
 mCam=CamOrientGenFromFile(mOriFileName,aICNM);
}
//===============================================================================//
/*                                Main Function                                  */
//===============================================================================//

int RegTIRVIS_main( int argc, char ** argv )
{

	// Call Elise Librairy We will be using because classes that handle 
	// These transformations are already computed 

        std::string Image_Pattern, Oris_VIS_dir,TestDataDIRpat, PlyFileIn;
		std::string Option;
        bool aTif=false;
		
    /************************************************************************/
     //Initilialize ElInitArgMain which as i understood captures arguments entered
     //by the operator
	/**********************************************************************/
		ElInitArgMain
		(
			argc,argv,
            LArgMain() << EAMC(TestDataDIRpat ,"Set of images to be used to estimate the homography",eSAM_IsPatFile)
                       << EAMC(Image_Pattern," Images pattern : Dir+ Name: prefixes Thermal: TIR, Visual: VIS", eSAM_IsPatFile)
                       << EAMC(Oris_VIS_dir," Orientation files for visual images",eSAM_IsExistDirOri)
                       << EAMC(PlyFileIn, " Ply file that will be used as a 3d model",eSAM_IsExistFile)
            ,LArgMain() << EAM(Option,"Option",true," FOR NOW, DON'T DO ANYTHING")
		);


/**********************************************************************/
	if (MMVisualMode) return EXIT_SUCCESS;




/*@@@@@@@@@@@@@@@@@@@@@@@@@ Train DATA IMAGES PROCESSING @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/




    //Take the testdata directory to compute the homography that
    //will be used as a predictor for keypoint matching

   // MakeFileDirCompl(TestDataDIRpat);
    //std::cout<<"Test data Directory "<<TestDataDIRpat<<endl;

    //=============iNITILIAZE CHANTIER Manipulateur========

    std::string aDirtest,aPatImTst;
    SplitDirAndFile(aDirtest,aPatImTst,TestDataDIRpat);


    std::cout<<"==============> Test data directory: "<<aDirtest<<endl;
    std::cout<<"==============> Pattern image test: "<<aPatImTst<<endl;


    // Chantier Manipulateur
    cInterfChantierNameManipulateur * aICNM0=cInterfChantierNameManipulateur::BasicAlloc(aDirtest);
    const std::vector<std::string> aSetImTest = *(aICNM0->Get(aPatImTst));
    std::cout<<"==============> Param Chantier manipulateur set \n";


/*===============================================================================*/
/*          Transform all images to 8 bits: Lab and wallis on 8 bits             */
/*===============================================================================*/
     std::size_t found = aPatImTst.find_last_of(".");
     string ext = aPatImTst.substr(found+1);
     cout<<"==============> Test images Extension : "<<ext<<endl;
     list<string> cmd;
     std::cout<<aSetImTest.size()<<endl;
    /* for (uint aK=0; aK<aSetImTest.size(); aK++)
      {
         std::cout<<aSetImTest[aK].c_str()<<endl;
      Tiff_Im Im=Tiff_Im::UnivConvStd(aSetImTest[aK]);
      if(Im.type_el()!=GenIm::u_int1)
        {
           std::cout<<"here \n";
           string aCmd = MM3DStr +  " To8Bits "+  aSetImTest[aK] + std::string(" AdaptMinMax=1");
            cmd.push_back(aCmd);
        }
      }
      cEl_GPAO::DoComInParal(cmd);

     std::cout<<"to 8bits done \n";*/

     // Test Images are set to 16 bits
     //cmd.clear();// CLEAR CMD to fill it again with the set of work images


//===============================================================================//
//                  Instantiate the detector MSD                                 //
/*===============================================================================*/
  MsdDetector msd;

  msd.setPatchRadius(3);
  msd.setSearchAreaRadius(5);

  msd.setNMSRadius(5);
  msd.setNMSScaleRadius(0);

  msd.setThSaliency(0.02);
  msd.setKNN(10);

  msd.setScaleFactor(1.25f);
  msd.setNScales(-1);

  msd.setComputeOrientation(true);
  msd.setCircularWindow(true);
  msd.setRefinedKP(false);

//===============================================================================//
/*      Rearrange the test images in couples of thermal and visual images        */
//===============================================================================//


/* Arrange images so that thermal images and visual images are stored in separate vectors*/

     std::vector< std::string > ThermalImages, VisualImages;
     std::size_t found_Vis;
     std::size_t found_Tir;
     std::size_t found_commonPattern;
     for (unsigned int j=0; j<aSetImTest.size(); j++)
         {
             found_Vis = aSetImTest[j].find("Vis");
             found_Tir = aSetImTest[j].find("Tir");
             if (found_Vis!=std::string::npos)
             {
                 VisualImages.push_back(aDirtest+aSetImTest[j]);
             }
             else
             {
                 if (found_Tir!=std::string::npos)
                 {
                     ThermalImages.push_back(aDirtest+aSetImTest[j]);
                 }
             }
         }


     std::cout<<"==============> Images are identified and arranged \n";

//===============================================================================//
/*  Sort the Thermal Images and Visual Images containers for ease of manipulation later*/
//===============================================================================//

     std::sort(VisualImages.begin(),VisualImages.end());
     std::sort(ThermalImages.begin(), ThermalImages.end());


     //Get Visual Image size
     /****************************************************/
     Tiff_Im Image=Tiff_Im::UnivConvStd(VisualImages[0]);
     Pt2di VisualSize(Image.sz());
     /****************************************************/


     //Create directory to store keypoints
     ELISE_fp::MkDirSvp(aDirtest + "KpsTEST/");
     string Path=aDirtest + "KpsTEST/";

     cElComposHomographie Ix(0,0,0);
     cElComposHomographie Iy(0,0,0);
     cElComposHomographie Iz(0,0,0);
     cElHomographie Hout(Ix,Iy,Iz);

     // Specify search parameters for the QuadTree structure
    /*********************************************************/
     FPRIMTp Pt_of_Point;
     Box2dr box(Pt2dr(0,0), Pt2dr(VisualSize.x,VisualSize.y) );
     ElSTDNS set<pair<int,Pt2dr> > Voisins ; // where to put nearest neighbours
     /*********************************************************** */


     std::cout<<"******************************************************************************\n"<<
                 "Computing the Homography Predictor using the test set with high variability  \n"<<
                "*******************************************************************************\n";
     for (unsigned int i=0; i<VisualImages.size();i++)
     {
         std::cout<<"enter regex\n";
          cElRegex rgx("VIS_(.*).tif",10);
          cElRegex rgxx("Vis_(.*).tif",10);
          cElRegex rgxxx("vis_(.*).tif",10);
          std::string aNameMatch;
          bool rgxMatch,rgxxMatch,rgxxxMatch;
          rgxMatch=rgx.Match(VisualImages[i]);
          rgxxMatch=rgxx.Match(VisualImages[i]);
          rgxxxMatch=rgxxx.Match(VisualImages[i]);
          if (rgxMatch||rgxxMatch||rgxxxMatch)
             {
              if(rgxMatch) aNameMatch=rgx.KIemeExprPar(1);
              if(rgxxMatch) aNameMatch=rgxx.KIemeExprPar(1);
              if(rgxxxMatch) aNameMatch=rgxxx.KIemeExprPar(1);
              std::cout<<"==============> Number of found images: "<<aNameMatch.c_str()<<endl;
             }
          found_commonPattern=ThermalImages[i].find(aNameMatch);

          if (found_commonPattern!=std::string::npos)
             {
              //Process the Termal Image: MSD --> Lab --> Wallis --> SIFT Descriptor
              Tiff_Im ImageV=Tiff_Im::UnivConvStd(VisualImages[i]);

              /**************************************************************************/
              // Check if MSD keypoints have already been computed: No need to do that again
              std::vector<KeyPoint> KpsV;
              string directory, file;
              SplitDirAndFile(directory,file,VisualImages[i]);
              string filenameV=Path + file + ".txt";


              if (DoesFileExist(filenameV.c_str()))
              {
                  std::cout<<"==============> Keypoints are already computed for image: "<<VisualImages[i]<<"\n";
                  Readkeypoints(KpsV, filenameV);
                  std::cout<<"Number of extracted MSD points: "<<KpsV.size()<<endl;
              }
              else
              {
                  std::cout<<"==============> Compute Keypoints for test image: "<<VisualImages[i]<<"\n";
                  KpsV=msd.detect(ImageV);
                  std::cout<<"Number of extracted MSD points: "<<KpsV.size()<<endl;
                  StoreKps(KpsV,filenameV);

              }
              /**************************************************************************/


              Im2D<U_INT1,INT> LabWImageV=Im2D<U_INT1,INT>(ImageV.sz().x,ImageV.sz().y);
             // std::cout<<"Before wallis \n";
              Migrate2Lab2wallis(ImageV,LabWImageV);

              //store image
              /*ELISE_COPY
              (
                  LabWImageV.all_pts(),
                  LabWImageV.in(),
                  Tiff_Im(
                      "LabWvisible.tif",
                      LabWImageV.sz(),
                      GenIm::u_int1,
                      Tiff_Im::No_Compr,
                      Tiff_Im::BlackIsZero,
                      Tiff_Im::Empty_ARG ).out()
              );*/


              //std::cout<<"after wallis and lab\n";
              //compute descriptors for visual
              DescriptorExtractor<U_INT1,INT> SIFTV=DescriptorExtractor<U_INT1,INT>(LabWImageV);
              vector<DigeoPoint> ListV;
              std::vector<KeyPoint>::iterator aKp=KpsV.begin();
              for (;aKp!=KpsV.end();++aKp)
              {
                  DigeoPoint DP;
                  DP.x=aKp->getPoint().x;
                  DP.y=aKp->getPoint().y;
                  REAL8  descriptor[DIGEO_DESCRIPTOR_SIZE];
                  DP.addDescriptor(0.0); // Add 0 angle
                  SIFTV.describe(DP.x,DP.y,aKp->getSize(),DP.angle(0),descriptor);
                  SIFTV.normalize_and_truncate(descriptor);
                  //std::cout<<"Descp   "<<descriptor[50]<<endl;
                  DP.addDescriptor(descriptor);
                  ListV.push_back(DP);
              }
              DigeoPoint::writeDigeoFile("DigeoV.txt",ListV);


              //Process the Termal Image: MSD --> Lab --> Wallis --> SIFT Descriptor
              Tiff_Im ImageTh=Tiff_Im::UnivConvStd(ThermalImages[i]);

              // Check if MSD keypoints have already been computed: No need to do that again
              SplitDirAndFile(directory,file,ThermalImages[i]);
              std::vector<KeyPoint> KpsTh;
              string filenameTh=Path + file + ".txt";

              if (DoesFileExist(filenameTh.c_str()))
              {
                  std::cout<<"==========> Keypoints are already computed for image: "<<ThermalImages[i]<<"\n";
                  Readkeypoints(KpsTh, filenameTh);
                  std::cout<<"Number of extracted MSD points: "<<KpsTh.size()<<endl;
              }
              else
              {
                  std::cout<<"===========> Compute Keypoints for test image: "<<ThermalImages[i]<<"\n";
                  KpsTh=msd.detect(ImageTh);
                  std::cout<<"Number of extracted MSD points: "<<KpsTh.size()<<endl;
                  StoreKps(KpsTh,filenameTh);

              }
              Im2D<U_INT1,INT> LabWImageTh=Im2D<U_INT1,INT>(ImageTh.sz().x,ImageTh.sz().y);
              Migrate2Lab2wallis(ImageTh,LabWImageTh);
              //store image
             /* ELISE_COPY
              (
                  LabWImageTh.all_pts(),
                  LabWImageTh.in(),
                  Tiff_Im(
                      "LabWThermal.tif",
                      LabWImageTh.sz(),
                      GenIm::u_int1,
                      Tiff_Im::No_Compr,
                      Tiff_Im::BlackIsZero,
                      Tiff_Im::Empty_ARG ).out()
              );*/
              DescriptorExtractor<U_INT1,INT> SIFTTh=DescriptorExtractor<U_INT1,INT>(LabWImageTh);
              vector<DigeoPoint> ListTh;

              aKp=KpsTh.begin();
              for (;aKp!=KpsTh.end();++aKp)
              {
                  DigeoPoint DP;
                  DP.x=aKp->getPoint().x;
                  DP.y=aKp->getPoint().y;
                  REAL8  descriptor[DIGEO_DESCRIPTOR_SIZE];
                  DP.addDescriptor(0.0);// add 0 angle
                  SIFTTh.describe(DP.x,DP.y,aKp->getSize(),DP.angle(0),descriptor);
                  SIFTTh.normalize_and_truncate(descriptor);
                  //std::cout<<"Descp   "<<descriptor[50]<<endl;
                  DP.addDescriptor(descriptor);
                  ListTh.push_back(DP);
              }

              DigeoPoint::writeDigeoFile("DigeoTH.txt",ListTh);
              list<string> cmd;
              string aCmd=MM3DStr +  " Ann "+ std::string("-ratio 0.9") + std::string(" DigeoTH.txt") + std::string(" DigeoV.txt") + std::string(" Matches.txt");
              cmd.push_back(aCmd);
              cEl_GPAO::DoComInParal(cmd);

              //Compute a Robust Homography out of the resulting Matching FIle
              ElPackHomologue HomologousPts;

              bool Exist= ELISE_fp::exist_file("Matches.txt");
              if (Exist)
                  {
                     HomologousPts= ElPackHomologue::FromFile("Matches.txt");
                  }
              cElComposHomographie Ix(0,0,0);
              cElComposHomographie Iy(0,0,0);
              cElComposHomographie Iz(0,0,0);
              cElHomographie H2estimate(Ix,Iy,Iz);

               double anEcart,aQuality;
               bool Ok;
               Hout=H2estimate.RobustInit(anEcart,&aQuality,HomologousPts,Ok,50,80.0,2000);

               std::cout<< " =================> Initial Homography  <=====================\n";
               Hout.Show();
               std::cout<<"Quality parameters \n";
               std::cout<<"========= >Ecart "<<anEcart<<endl;
               std::cout<<"========= >Quality "<<aQuality<<endl;
               std::cout<<"========= >If Ok= 1 " <<Ok<<endl;

               ArbreKD * ArbreV= new ArbreKD(Pt_of_Point, box, KpsV.size(), 1.0);
               for (uint i=0; i<KpsV.size(); i++) {
                   ArbreV->insert(pair<int,Pt2dr>(i, Pt2dr(KpsV.at(i).getPoint().x, KpsV.at(i).getPoint().y)));
               }

               // Call Enrich Keypoints to use the homography as predictor
               EnrichKps(KpsTh,ArbreV,Hout,5);

               delete ArbreV;
             }
     }

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/







/*@@@@@@@@@@@@@@ DYKE DATA SET MATCHING USING HOMOGRAPHY PREDICTOR @@@@@@@@@@@@@@@@@@*/

     msd.setCircularWindow(false); //Bach to square like patch


//===============================================================================//
  /*   MakeFileDirCompl(Oris_VIS_dir);
     std::cout<<"Oris_Dir dir: "<<Oris_VIS_dir<<std::endl;*/

/*********************************************************************************/
    // Initialize name manipulator & files :: now with work images
	std::string aDirImages,aPatIm;
	SplitDirAndFile(aDirImages,aPatIm,Image_Pattern);
	std::cout<<"Working dir: "<<aDirImages<<std::endl;
	std::cout<<"Images pattern: "<<aPatIm<<std::endl;
	
/*********************************************************************************/

 cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
 const std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));
 
//===============================================================================//
 //Transform all images to 16 bits
 //===============================================================================//

 found = Image_Pattern.find_last_of(".");
 ext   = Image_Pattern.substr(found+1);
 cout<<"Ext : "<<ext<<endl;

 if ( ext.compare("tif"))   //ext equal tif
 {
     aTif = true;
     cout<<"Tif images"<<endl;
 }
 if (aTif)
 {
     list<string> cmd;
     for (uint aK=0; aK<aSetIm.size(); aK++)
     {
          string aCmd = MM3DStr +  " PastDevlop "+  aSetIm[aK] + " Sz1=-1 Sz2=-1 Coul8B=0";
          cmd.push_back(aCmd);
     }
     cEl_GPAO::DoComInParal(cmd);
 }


 /* Arrange images so that thermal images and visual images are stored in separate vectors*/

  ThermalImages.clear(); VisualImages.clear();
 for (unsigned int j=0; j<aSetIm.size(); j++)
     {
         found_Vis = aSetIm[j].find("VIS");
         found_Tir = aSetIm[j].find("TIR");
         if (found_Vis!=std::string::npos)
         {
             VisualImages.push_back(aSetIm[j]);
         }
         else
         {
             if (found_Tir!=std::string::npos)
             {
                 ThermalImages.push_back(aSetIm[j]);
             }
         }
     }




//===============================================================================//
 // Sort the Thermal Images and Visual Images containers for ease of manipulation later
 //===============================================================================//

 std::sort(VisualImages.begin(),VisualImages.end());
 std::sort(ThermalImages.begin(), ThermalImages.end());



/***********************************************************************************/
 // Apply the homography to the set of thermal images and store images in a folder named Thermal Homography applied

/*************************************************/
 /*std::string DirectoryRect= "./RectThermalImages";
 if (!ELISE_fp::IsDirectory(DirectoryRect))
 {
     ELISE_fp::MkDir(DirectoryRect);
 }
 for (uint i=0; i<ThermalImages.size();i++)
 {
     std::string imm=DirectoryRect + ThermalImages[i]+"_Rec.tif";
     ApplyHomography(ThermalImages[i],VisualImages[i],Hout,imm);
 }*/
 /**********************************************************************/

//===============================================================================//
// PROCESS THermal by applying the computed homography and take advantage of the
// RGB orientation
//===============================================================================//

 //Steps:
 //1. GrapheHom to get possible OVERLAPPING relations between images
 //2. Apply TiPunch to get RGB mesh file
 //3. ZBufferRaster to get depth images
 //4. Process to compute homologous points

 //Compute the graph of image correspondences
 list< string > CMD;
 string aCMD;

 //Define the pattern of visual images
 string aPatImVIS="VIS*"+aPatIm;

 //1. Read the xml file containing couples of images
std::cout<<"=================> Computing optical images overlapping relations: Visibility graph\n";
std::vector< cCpleString> ImCpls;
if (!DoesFileExist("GrapheHom.xml"))
{
    aCMD= MM3DStr + " GrapheHom " + aDirImages + " \"" + aPatImVIS + "\" " + Oris_VIS_dir;
    //std::cout<<" Command "<<aCMD<<endl;
    CMD.push_back(aCMD);
    cEl_GPAO::DoComInParal(CMD);

    std::cout<<"=================> Visibility graph built \n";
}



 if (DoesFileExist("GrapheHom.xml"))
 {
    cSauvegardeNamedRel aRel=StdGetFromPCP("GrapheHom.xml",SauvegardeNamedRel);
    ImCpls=aRel.Cple();
    /*for (uint i=0;i<ImCpls.size();i++)
    {
        std::cout<<ImCpls.at(i).N1()<< " "<<ImCpls.at(i).N2()<<endl;
    }*/
 }
 else
 {
     std::cout<<"There is no homol graph that will be used to compute Homologous points\n";
 }


 CMD.clear();


 //2. Apply TiPunch to get RGB mesh file
 /*aCMD=MM3DStr + " TiPunch " + PlyFileIn + " Pattern=\"" + aPatImVIS + "\" " +  "Out=MeshCloud.ply " + "Mode=MicMac";

 std::cout<<aCMD<<endl;
CMD.push_back(aCMD);
cEl_GPAO::DoComInParal(CMD);
 std::cout<<aCMD<<endl;*/


ELISE_ASSERT(ELISE_fp::exist_file(PlyFileIn),"Mesh file not computed or is corrupt ");


//3. Compute ZBuffer Images to get depth information for each pixel
string dirDepthImages="./Tmp-ZBuffer/" + PlyFileIn;

if (!ELISE_fp::IsDirectory(dirDepthImages))
{
    aCMD=MM3DStr + " TestLib ZBufferRaster " + PlyFileIn + " \"" + aPatImVIS + "\" " + Oris_VIS_dir;
    CMD.push_back(aCMD);
    cEl_GPAO::DoComInParal(CMD);
}

ELISE_ASSERT(ELISE_fp::IsDirectory("./Tmp-ZBuffer"), "ZBuffer Directory has not been created");



//===============================================================================//
// Compute MSD keypoints for all images and store them in Keypoints directory
//===============================================================================//

ELISE_ASSERT(ThermalImages.size()==VisualImages.size(),"IR and RGB images have not the same count ");
std::string  Kpsfile="./Keypoints_MSD";
if (!ELISE_fp::exist_file(Kpsfile))
{
    ELISE_fp::MkDir(Kpsfile);
}

// Check also if Keypoints contains file for kps

for (uint i=0;i<ThermalImages.size();i++)
    {
        std::string FileTh=Kpsfile + "/" + ThermalImages.at(i) + ".txt";
        if (!DoesFileExist(FileTh.c_str()))
        {
            Tiff_Im ImTh=Tiff_Im::UnivConvStd(ThermalImages.at(i));
            std::vector<KeyPoint> KpsTh=msd.detect(ImTh);
            std::cout<<"Number of extracted MSD points for image: "<<ThermalImages[i]<<KpsTh.size()<<endl;
            StoreKps(KpsTh,FileTh);
        }

        std::string FileV=Kpsfile + "/" + VisualImages.at(i) + ".txt";
        if(!DoesFileExist(FileV.c_str()))
        {

            Tiff_Im ImV=Tiff_Im::UnivConvStd(VisualImages.at(i));
            std::vector<KeyPoint> KpsV=msd.detect(ImV);
            std::cout<<"Number of extracted MSD points for image: "<<VisualImages[i]<<KpsV.size()<<endl;
            StoreKps(KpsV,FileV);
        }
    }

//===============================================================================//
// Compute SIFT keypoints for all images and store them in Keypoints directory
//===============================================================================//

ELISE_ASSERT(ThermalImages.size()==VisualImages.size(),"IR and RGB images have not the same count ");
std::string KpsfileSIFT="./Keypoints_SIFT";
if (!ELISE_fp::IsDirectory(KpsfileSIFT))
{
    ELISE_fp::MkDir(KpsfileSIFT);
}

// Check also if Keypoints contains a file for kps
CMD.clear();
for (uint i=0;i<ThermalImages.size();i++)
    {
        std::string FileTh=KpsfileSIFT + "/" + ThermalImages.at(i) + ".key";
        if /*(!DoesFileExist(FileTh.c_str()))*/ (!ELISE_fp::exist_file(FileTh))
        {
            aCMD= MM3DStr + " Sift " + ThermalImages.at(i) + " -o " + FileTh;
            CMD.push_back(aCMD);
        }
        std::string FileV=KpsfileSIFT + "/" + VisualImages.at(i) + ".key";
        if /*(!DoesFileExist(FileV.c_str()))*/ (!ELISE_fp::exist_file(FileV))
        {
            aCMD= MM3DStr + " Sift " + VisualImages.at(i) + " -o " + FileV;
            CMD.push_back(aCMD);
        }
    }
cEl_GPAO::DoComInParal(CMD);
//std::cout<<"Sift KeyPoints computed for all images \n";


//===============================================================================//
// Apply the homography to each image of the thermal images
//===============================================================================//



// IR/IR registration: Homol folder contains all correspondences
std::string  Homolfile="./Homol";
std::vector< KeyPoint > Kps1,Kps2;  // MSD KeyPoints
std::vector< SiftPoint > Kps1S,Kps2S; // SIFT Keypoints
std::vector<KeyPoint> Kps1Skp; // move SiftPoint to KeyPoint frame
std::vector<KeyPoint> Kps2Skp; // move SiftPoint to KeyPoint frame
std::vector<Pt2dr> Kps1H;
std::vector<Pt2dr> Kps2H;
std::vector<Pt2dr> Kps1HS;
std::vector<Pt2dr> Kps2HS;
ElPackHomologue HomologousPts;
// Instantiate a Depth image that is to be used for reprojection
Im2D<REAL4,REAL> Depth=Im2D<REAL4,REAL>();

if (!ELISE_fp::IsDirectory(Homolfile)) // Homolfile is not created
{
    std::cout<<"********************************************************************************************\n"<<
               "*********************************************************************************************\n"<<
               "     Computing thermal tie points using the homography predictor and the 3D optical model    \n"<<
               "*********************************************************************************************\n"<<
               "**********************************************************************************************\n";

    ELISE_fp::MkDir(Homolfile); // create this directory

    for (uint i=0;i<ImCpls.size();i++)
    {
        //Clear all containers
        Kps1.clear();Kps1S.clear();Kps2.clear();Kps2S.clear();
        Kps1Skp.clear();Kps2Skp.clear();
        Kps1H.clear();Kps2H.clear();Kps1HS.clear();Kps2HS.clear();

        // Clear the set of homologous points
        HomologousPts.clear();


        string ImageTh1=WhichThermalImage(ImCpls.at(i).N1(),ThermalImages);
        string ImageTh2=WhichThermalImage(ImCpls.at(i).N2(),ThermalImages);
        std::cout<<" Couple of thermal images  "<<ImageTh1<<"        "<<ImageTh2<<endl;
        std::string DirPastis=Homolfile + "/Pastis" + ImageTh1;


        if (!ELISE_fp::exist_file(DirPastis))
        {
            //create directory Pastis
            ELISE_fp::MkDir(DirPastis);
        }



        string file12=DirPastis + "/" + ImageTh2 + ".txt";
        if (!ELISE_fp::exist_file(file12))
        {


            //First thermal image: get Keypoints ==> MSD
            /******************************************************/
            string filecpleIm1=Kpsfile + "/" + ImageTh1 + ".txt";
            Readkeypoints(Kps1,filecpleIm1);
            /*******************************************************/
            //First thermal image: get Keypoints ==> SIFT
            /*******************************************************/
            string filecpleIm1S= KpsfileSIFT + "/" + ImageTh1 +".key";
            read_siftPoint_list(filecpleIm1S,Kps1S);
            /*******************************************************/
            // Second thermal image: get Keypoints ==> MSD
            /*******************************************************/
            string filecpleIm2=Kpsfile + "/"+ImageTh2 + ".txt";
            Readkeypoints(Kps2,filecpleIm2);
            /*******************************************************/
            // Second thermal image: get Keypoints ==> SIFT
            /*******************************************************/
           string filecpleIm2S= KpsfileSIFT + "/" + ImageTh2+ ".key";
           read_siftPoint_list(filecpleIm2S,Kps2S);
           /*******************************************************/
            // Apply the computed homography to all set of keypoints

           /****************MSD MSD MSD MSD MSD ******************/
            Kps1H=NewSetKpAfterHomog(Kps1,Hout);
            Kps2H=NewSetKpAfterHomog(Kps2,Hout);
            /***************SIFT SIFT SIFT SIFT SIFT *************/
            Kps1HS=NewSetKpAfterHomog(Kps1S,Hout);
            Kps2HS=NewSetKpAfterHomog(Kps2S,Hout);


            std::cout<<"MSD Points "<<Kps1H.size()<<"        "<<Kps2H.size()<<endl;
            std::cout<<"SIFT Points "<<Kps1HS.size()<<"        "<<Kps2HS.size()<<endl;


            //At this step, we can merge SIFT and MSD keypoints

            Kps1Skp= FromSiftP2KeyP(Kps1S); // move to KeyPoint frame
            Kps2Skp= FromSiftP2KeyP(Kps2S); // move to KeyPoint frame

            Kps1.insert(Kps1.end(),Kps1Skp.begin(),Kps1Skp.end());
            Kps2.insert(Kps2.end(),Kps2Skp.begin(),Kps2Skp.end());


            Kps1H.insert(Kps1H.end(),Kps1HS.begin(),Kps1HS.end());
            Kps2H.insert(Kps2H.end(),Kps2HS.begin(),Kps2HS.end());




            //Use the oriented Visual Images to compute image1==>Terrain=>image2
            Orient_Image ImV1(Oris_VIS_dir,ImCpls.at(i).N1(),aICNM);
            Orient_Image ImV2(Oris_VIS_dir,ImCpls.at(i).N2(),aICNM);


            //Apply Image1 ==> Terrain operation having into consideration a depth image
            // from ZBufferRaster directory
            string FileDepthImV1="./Tmp-ZBuffer/" + PlyFileIn + "/" + ImCpls.at(i).N1() + "/" + ImCpls.at(i).N1() + "_ZBuffer_DeZoom1.tif";
            ELISE_ASSERT(DoesFileExist(FileDepthImV1.c_str()),"The Depth image relative to visual Image is not found");


            //Creating and filling the KDTree structure
            /*********************************************************************/
            ArbreKD * SlaveTree= new ArbreKD(Pt_of_Point, box, Kps2H.size(), 1.0);
            for (uint l=0; l<Kps2H.size(); l++) {
                SlaveTree->insert(pair<int,Pt2dr>(l, Kps2H.at(l)));
            }
            /*********************************************************************/


            Depth=Im2D<REAL4,REAL>::FromFileStd(FileDepthImV1);

            for (uint j=0;j<Kps1H.size();j++)
            {
                // Sift repetetive points are caused by scale space definition, onr should filter theses occurences
                // so that one among them is only processed
                if (j>0 && (Kps1H.at(j).x==Kps1H.at(j-1).x && Kps1H.at(j).y==Kps1H.at(j-1).y))
                {
                    continue;
                }

                double Prof;
                // Added Condition due to an error of segmantation: point where depth is not defined
                /*************************************************************************************/
                if (IncludedInDepthIm(Kps1H.at(j),Depth.sz()))
                {
                    Prof=Depth.BilinIm()->Get(Kps1H.at(j));
                }
                else
                {
                    Prof=-1;
                }
                /**************************************************************************************/


                if (Prof!=-1)
                {

                    Pt3dr pTerrain= ImV1.getCam()->ImDirEtProf2Terrain(Kps1H.at(j),Prof,ImV1.getCam()->DirVisee());
                    Pt2dr PtImage2= ImV2.getCam()->Ter2Capteur(pTerrain);

                    Voisins.clear();
                    SlaveTree->voisins(PtImage2, distMax, Voisins);

                    if (Voisins.size()>0)
                     {
                        Pt2dr P1(Kps1.at(j).getPoint().x, Kps1.at(j).getPoint().y);
                        Pt2dr Pnew(Kps2.at(Voisins.begin()->first).getPoint().x,Kps2.at(Voisins.begin()->first).getPoint().y);
                        HomologousPts.Cple_Add(ElCplePtsHomologues(P1,Pnew));
                     }
                }
            }
            delete SlaveTree;
            Depth.raz();
            HomologousPts.StdPutInFile(file12);
        }


    }
}
else
{
    std::cout<<"********************************************************************************************\n"<<
               "*********************************************************************************************\n"<<
               "        Thermal tie points are already computed and there is a filled ./Homol directory      \n"<<
               "*********************************************************************************************\n"<<
               "**********************************************************************************************\n";
}


//===============================================================================//
// Trying to estimate a 3D similarity transform: 3 Rotations, 3 Translations and
// a Scale factor from the the set of inter-modality homologous points
//===============================================================================//



std::cout<<"********************************************************************************************\n"<<
           "*********************************************************************************************\n"<<
            "Computing thermo-optical tie points using the homography predictor and the 3D optical model \n"<<
            "   ===> Estimation of a 3D similarity transform between thermal and optical coordinate systems\n"<<
           "*********************************************************************************************\n"<<
           "**********************************************************************************************\n";
//Steps to be followed
/*
 * 1. For each thermal image, try to parse the graph homol file generated by GrapheHom
 * 2. For all possibble corrspondence images
 *    Reproject the 3D point got from the theraml image
 *    Check if there are correspondences
 *    If there is a corrspondence
 *       Commence constructing the structure of mesure appuis file <SetOfMesureAppuisFlottants> with an incremental naming convention
 *       and accordingly fill the 3D measure xml file.
 */

/*std::vector< Im2D<U_INT2,INT>* > *AllMasks= new std::vector<Im2D<U_INT2,INT> *>;*/
std::vector< std::vector<KeyPoint> > *AllKps= new std::vector< std::vector<KeyPoint> >;
std::vector<std::vector<Pt2dr> > *AllKpsThermalHomog= new std::vector<std::vector<Pt2dr> >;


std::vector< ArbreKD* >  * AllTrees = new std::vector < ArbreKD* >;
//std::vector< std::vector<KeyPoint>  > *AllthermalkpsBeforeH= new std::vector< std::vector<KeyPoint> >;
//Define a class that is compatible to the xml 2D MEASURE file


// Clear Depth image
/*********************************************************/

 Depth.raz();

/*********************************************************/


// Fill all containers

 std::cout<<"IMAGE         ||  MSD POINTS  ||  SIFT POINTS \n";

for (uint i=0; i<ThermalImages.size();i++)
{

    //Clear all containers
    Kps1.clear();Kps1S.clear();// kps MSD + SIFT

    Kps2.clear();Kps2S.clear(); // KPS Msd + SIFT

    Kps1Skp.clear();Kps2Skp.clear(); // Used to move from SiftPOint 2 KeyPoint


    Kps1H.clear();Kps1HS.clear(); // Applying Homography to Msd Kps Kps1 and SIFT kps Kps1S*/

    //Thermal image: get KeyPoints ==> MSD
    /******************************************************/
    string filecpleIm1=Kpsfile + "/" + ThermalImages[i] + ".txt";
    Readkeypoints(Kps1,filecpleIm1);
    /*******************************************************/
    //Thermal image: get KeyPoints ==> SIFT
    /*******************************************************/
    string filecpleIm1S= KpsfileSIFT + "/" + ThermalImages[i] +".key";
    read_siftPoint_list(filecpleIm1S,Kps1S);
    std::cout<<ThermalImages[i]<<" ||  "<<Kps1.size()<<"  ||  "<<Kps1S.size()<<endl;
    /*******************************************************/
    // Visual image: get KeyPoints ==> MSD
    /*******************************************************/
    string filecpleIm2=Kpsfile + "/"+VisualImages[i] + ".txt";
    Readkeypoints(Kps2,filecpleIm2);
    /*******************************************************/
    // Visual image: get KeyPoints ==> SIFT
    /*******************************************************/
    string filecpleIm2S= KpsfileSIFT + "/" + VisualImages[i]+ ".key";
   read_siftPoint_list(filecpleIm2S,Kps2S);
   std::cout<<VisualImages[i]<<" ||  "<<Kps2.size()<<"  ||  "<<Kps2S.size()<<endl;
   /*******************************************************/

   // Apply homography only to thermal image KeyPoints: MSD +SIFT
   /****************MSD MSD MSD MSD MSD ******************/
    Kps1H=NewSetKpAfterHomog(Kps1,Hout);
    /***************SIFT SIFT SIFT SIFT SIFT *************/
    Kps1HS=NewSetKpAfterHomog(Kps1S,Hout);


    //At this step, we can merge SIFT and MSD KeyPoints

    Kps1Skp= FromSiftP2KeyP(Kps1S); // move to KeyPoint frame
    Kps2Skp= FromSiftP2KeyP(Kps2S); // move to KeyPoint frame

    Kps1.insert(Kps1.end(),Kps1Skp.begin(),Kps1Skp.end());// KeyPoints MSD+SIFT in the original thermal image SPACE
    Kps1H.insert(Kps1H.end(),Kps1HS.begin(),Kps1HS.end());// KeyPoints of thermal image after applying homography

    Kps2.insert(Kps2.end(),Kps2Skp.begin(),Kps2Skp.end());// KeyPoints of visual image no changes are done just concatenate MSD and SIFT


    // Instead, we want to create QuadTrees to search for corresponding tie points


    ArbreKD * ArbreTH= new ArbreKD(Pt_of_Point, box, Kps1H.size(), 1.0);
    for (uint i=0; i<Kps1H.size(); i++) {
        ArbreTH->insert(pair<int,Pt2dr>((int)i, Kps1H.at(i)));
    }
    AllTrees->push_back(ArbreTH);

    ArbreKD * ArbreV= new ArbreKD(Pt_of_Point, box, Kps2.size(), 1.0);
    for (uint i=0; i<Kps2.size(); i++) {
        ArbreV->insert(pair<int,Pt2dr>((int)i, Pt2dr(Kps2.at(i).getPoint().x, Kps2.at(i).getPoint().y)));
    }
    AllTrees->push_back(ArbreV);
    AllKps->push_back(Kps1);
    AllKps->push_back(Kps2);
    AllKpsThermalHomog->push_back(Kps1H);
}

std::cout<<"********************************************************************************************\n"<<
           "*********************************************************************************************\n"<<
            "                         All Trees are build for the searching process \n"<<
           "*********************************************************************************************\n"<<
           "**********************************************************************************************\n";

// Now, consider one image, use Graph homol to see what images are seen.
// Reproject the 3D point relative to every KeyPoint ray on these images, check if there are KeyPoints
// Store these KeyPoints under the same name in An xml file named Mesure-2D.xml equivalent to that computed
// by SaisieAppuisInit

std::vector< int > WhichIsSeen;
WhichIsSeen.resize(ThermalImages.size()*2);

std::vector< int >  MaskFoundKps;
MaskFoundKps.resize(ThermalImages.size()*2);



// define all sets to store 2d points and 3d ground truth points

cSetOfMesureAppuisFlottants AllSet; // Is the class that encapsulates all the set of appuis
std::vector< cMesureAppuiFlottant1Im > SetAppuis;
SetAppuis.resize(ThermalImages.size()*2);

// Mesure-3D.xml

cDicoAppuisFlottant All3Dpts;

// Instantiate Name Images

for(uint j=0; j<ThermalImages.size();j++)
{
    SetAppuis.at(2*j).NameIm()=ThermalImages[j];
    SetAppuis.at(2*j+1).NameIm()=VisualImages[j];
}

/************************************************************************************/

for (uint i=0;i< ThermalImages.size();i++)
{
    //  All what is done for a thermal image

    /*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/
    /*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/
    /* 1. Take a thermal image
     * 2. Which visual image that is homologous.
     * 3. Search inside ImCpls the images that can be seen from step 2.
     * 4. Create a Mask of length the size of visualImages with 0 and 1
     *    to index seen images with 1 ==> Useful later to search for corresponding
     *    mask of keypoints.
     * 5. Duplicate the resulting Mask to acount for thermal Images
     * 6. We obtain a Mask like 00 11 00 11 11 11 denoting that VisualImage 0 and thermalImage 0
     *    are not among the seen images
     *    "11" means that VisualImages[1] and ThermalImages[1] are in scope and we can use their corresponding
     *    masks of keypoints.
     */
    /*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/

    //search inside ImCpls
    std::vector<string> VisuHomols;
    ParseHomol(VisualImages[i],ImCpls,VisuHomols);
    std::sort(VisuHomols.begin(),VisuHomols.end());
    int c0=0;
    for(uint j=0;j<VisualImages.size();j++)
    {
        if (VisualImages[j].compare(VisuHomols[c0])!=0)
        {
            WhichIsSeen[2*j]=WhichIsSeen[2*j+1]=0;
        }
        else
        {
            WhichIsSeen[2*j]=WhichIsSeen[2*j+1]=1;
            if (c0<(int)VisuHomols.size()-1)
            {
               c0++;
            }
        }
    }
    // Add the Image that is homologous to the thermal image
    WhichIsSeen[2*i+1]=1; // its means that VisualImages[i] is added since it is seen by ThermalImages[i]
    //Now we have a mask that contains indexes to all kpsMasks that are involved in the calculation
    //ALWAYS couples of TIR/RGB


    std::cout<<"============>> Master thermal image:"<<ThermalImages[i]<<"\n";
    //Use the oriented Visual Images to compute image1==>Terrain=>image2
    Orient_Image ImV(Oris_VIS_dir,VisualImages[i],aICNM);



    //Apply Image ==> Terrain operation having into consideration a depth image
    // from ZBufferRaster directory

    string FileDepthImV="./Tmp-ZBuffer/" + PlyFileIn + "/" + VisualImages[i] + "/" + VisualImages[i] + "_ZBuffer_DeZoom1.tif";
    ELISE_ASSERT(DoesFileExist(FileDepthImV.c_str()),"The Depth image relative to visual Image is not found");
    //std::cout<<"============>> Depth image for image: "<<VisualImages[i]<<" created  \n";


    // load Depth image corresponding to the visual image needed for the task
    Depth=Im2D<REAL4,REAL>::FromFileBasic(FileDepthImV);

    std::vector< ImAndPt > Multiplicity; // We seek multiple correspondences between thermal and optical images
                                         // to compute the 3D similarity

/*========================================================================
*===============>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*========>>>>>>Check every keypoint of thermal images in loop< <<<<<<<<=====
*===============>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*========================================================================*/
    for (uint k=0;k<AllKpsThermalHomog->at(i).size();k+=150)// Until Now there is a bug whenever all the points are invloved
    {
        // Sift repetetive points are caused by scale space definition, one should filter theses occurences
        // so that one among them is only processed

        if (k>0 && (AllKpsThermalHomog->at(i).at(k).x==AllKpsThermalHomog->at(i).at(k-1).x
                    && AllKpsThermalHomog->at(i).at(k).y==AllKpsThermalHomog->at(i).at(k-1).y))
        {
            continue;
        }
        Multiplicity.clear();


        // Added Condition due to an error of segmantation: point where depth is not defined
        /*************************************************************************************/
        double Prof;

        if (IncludedInDepthIm(AllKpsThermalHomog->at(i).at(k), Depth.sz()))
        {
             Prof=Depth.BilinIm()->Get(AllKpsThermalHomog->at(i).at(k));
        }
        else
        {
            Prof=-1;
        }
        /*************************************************************************************/

        if (Prof!=-1)
        {
            //std::cout<<"The depth image is defined: Value= "<<Prof<<"\n";

            Pt3dr pTerrain= ImV.getCam()->ImDirEtProf2Terrain(AllKpsThermalHomog->at(i).at(k),Prof,ImV.getCam()->DirVisee());


           // Found keypoints will be at best in all viewable images

            for (uint l=0;l<ThermalImages.size()*2;l++)
             {
                 MaskFoundKps[l]=WhichIsSeen[l];
             }


            for (uint n=0;n<ThermalImages.size();n++)
            {
                if(WhichIsSeen[2*n]) // there is a couple of thermal and visual image seen by the master image
                {
                    Orient_Image ImVslave(Oris_VIS_dir,VisualImages[n],aICNM);
                    Pt2dr Ptslave= ImVslave.getCam()->Ter2Capteur(pTerrain);
                    Voisins.clear();
                    AllTrees->at(2*n)->voisins(Ptslave, distMax, Voisins);
                    if (Voisins.size()>0)
                    {
                        if (WithinLimits(Voisins.begin()->first, AllKps->at(2*n)))
                        {
                            Pt2dr YesHom(AllKps->at(2*n).at(Voisins.begin()->first).getPoint().x,AllKps->at(2*n).at(Voisins.begin()->first).getPoint().y);
                            ImAndPt ImPt={ThermalImages[n],YesHom};
                            Multiplicity.push_back(ImPt);
                        }
                        else
                        {
                            MaskFoundKps[2*n]=0;
                        }
                    }
                    else
                    {
                        MaskFoundKps[2*n]=0; // There is no found keypoint in this viewable image
                    }

                    Voisins.clear(); // Clear Voisins which contains the Nearest Neighbours to a point
                    AllTrees->at(2*n+1)->voisins(Ptslave, distMax, Voisins);
                    if (Voisins.size()>0)
                    {
                        if (WithinLimits(Voisins.begin()->first, AllKps->at(2*n+1)))
                        {
                            ImAndPt ImPt={VisualImages[n],Voisins.begin()->second};
                            Multiplicity.push_back(ImPt);
                        }
                        else
                        {
                            MaskFoundKps[2*n+1]=0;
                        }
                    }
                    else
                    {
                        MaskFoundKps[2*n+1]=0; // There is no found keypoint in this viewable image
                    }

                }
                else
                {
                    if (WhichIsSeen[2*n+1]) //Its homologous visual image
                    {

                        Orient_Image ImVslave(Oris_VIS_dir,VisualImages[n],aICNM);
                        Pt2dr Ptslave= ImVslave.getCam()->Ter2Capteur(pTerrain);
                        Voisins.clear();
                        AllTrees->at(2*n+1)->voisins(Ptslave, distMax, Voisins);

                        if (Voisins.size()>0)
                        {
                            if (WithinLimits(Voisins.begin()->first, AllKps->at(2*n+1)))
                            {
                                ImAndPt ImPt={VisualImages[n],Voisins.begin()->second};
                                Multiplicity.push_back(ImPt);
                            }
                            else
                            {
                                MaskFoundKps[2*n+1]=0;
                            }
                        }
                        else
                        {
                            MaskFoundKps[2*n+1]=0; // There is no found keypoint in this viewable image
                        }
                     }
                }
            }
            //check if point is multiple: seen more than three times in different images

            if (Multiplicity.size()>=4)
               {
                    //std::cout<<" There is a point which is multiple \n";

                    // define a name to the Appui Point
                    string Namept="Pt_ImTh"+ ToString((int)i)+"_"+ToString((int)k);

                    /**************************************************************/
                    // store 3d point coordinates in cDicoAppuisFlottants
                    cOneAppuisDAF Point3D;
                    Point3D.Pt()=pTerrain;
                    Point3D.NamePt()=Namept;
                    Pt3dr Incertitude(1,1,1);
                    Point3D.Incertitude()=Incertitude;
                    All3Dpts.OneAppuisDAF().push_back(Point3D);
                    /**************************************************************/

                    // Fill cMesureAppuisFlottants relative to master image

                    cOneMesureAF1I Measure;
                    Measure.NamePt()=Namept;
                    Measure.PtIm()=Pt2dr(AllKps->at(2*i).at(k).getPoint().x,AllKps->at(2*i).at(k).getPoint().y);
                    SetAppuis.at(2*i).OneMesureAF1I().push_back(Measure);

                    // Take a look at MaskFoundKps
                    uint iMul=0;
                    for (int p=0;p<(int)ThermalImages.size();p++)
                    {
                        if (MaskFoundKps[2*p]) // There is a keypoint in a thermal image
                        {
                            cOneMesureAF1I Measure;
                            Measure.NamePt()=Namept;
                            ELISE_ASSERT(ThermalImages[p]==Multiplicity.at(iMul).ImageName," Something went wrong in Multiplicity or naming conventions")
                            Measure.PtIm()=Multiplicity.at(iMul).MesureImage;
                            SetAppuis.at(2*p).OneMesureAF1I().push_back(Measure);
                            if(iMul<Multiplicity.size()-1){iMul++;}
                        }
                        if (MaskFoundKps[2*p+1])  // There is a KeyPoint in a visual image
                        {
                            cOneMesureAF1I Measure;
                            Measure.NamePt()=Namept;
                            ELISE_ASSERT(VisualImages[p]==Multiplicity.at(iMul).ImageName," Something went wrong in Multiplicity or naming conventions")
                            Measure.PtIm()=Multiplicity.at(iMul).MesureImage;
                            SetAppuis.at(2*p+1).OneMesureAF1I().push_back(Measure);
                            if(iMul<Multiplicity.size()-1){iMul++;}
                        }
                    }
               }

        }

    }

/*========================================================================
*===============>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*========>>>>>>Check every keypoint of Visual images in loop<<<<<<<<<======
*===============>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*========================================================================*/

    /*First of All,
      ==> Update Mask that embeds Visibility images
      */
    WhichIsSeen[2*i+1]=0; // Its our new master image ==> self Visibility ==> "awkward"
    WhichIsSeen[2*i]=1; //   Its homologous thermal image is included in visibility mask


    //Then Do almost the same thing
    std::cout<<"============>> Master optical image:"<<VisualImages[i]<<"\n";

    for (uint k=0;k<AllKps->at(2*i+1).size();k+=150)
    {
        // Sift repetetive points are caused by scale space definition, onr should filter theses occurences
        // so that one among them is only processed
        if (k>0 && ((AllKps->at(2*i+1).at(k).getPoint().x==AllKps->at(2*i+1).at(k-1).getPoint().x)
                    && (AllKps->at(2*i+1).at(k).getPoint().y==AllKps->at(2*i+1).at(k-1).getPoint().y)))
        {
            continue;
        }

        Multiplicity.clear();
        Pt2dr akPos(AllKps->at(2*i+1).at(k).getPoint().x,AllKps->at(2*i+1).at(k).getPoint().y);
        //std::cout<<"============> Keypoint for which a depth should be given: "<<akPos<<endl;

        double Prof;
        if (IncludedInDepthIm(akPos,Depth.sz()))
        {
           Prof=Depth.BilinIm()->Get(akPos);
        }
        else
        {
            Prof=-1;
        }

       // std::cout<<"============> Depth value: "<<Prof<<endl;

        if (Prof!=-1)
        {
            Pt3dr pTerrain= ImV.getCam()->ImDirEtProf2Terrain(akPos,Prof,ImV.getCam()->DirVisee());
            //std::cout<<"============> Point terrain: "<<pTerrain<<endl;
           // Found keypoints will be at best in all viewable images
            for (uint l=0;l<ThermalImages.size()*2;l++)
             {
                 MaskFoundKps[l]=WhichIsSeen[l];
             }
            for (uint n=0;n<VisualImages.size();n++)
            {
                if(WhichIsSeen[2*n+1]) // there is a couple of thermal and visual image seen by the master visual image
                {
                    Orient_Image ImVslave(Oris_VIS_dir,VisualImages[n],aICNM);
                    Pt2dr Ptslave= ImVslave.getCam()->Ter2Capteur(pTerrain);
                    //std::cout<<"============> Reprojection point: "<<Ptslave<<endl;
                    Voisins.clear(); // Clear Voisins which contains the Nearest Neighbours to a point
                    AllTrees->at(2*n)->voisins(Ptslave, distMax, Voisins);
                    if (Voisins.size()>0)
                    {
                        //std::cout<<"============> First index of neighbours: "<<Voisins.begin()->first<<"    Relevant keypoints size:  "<<AllKps->at(2*n).size()<<endl;
                        if (WithinLimits(Voisins.begin()->first, AllKps->at(2*n)))
                        {
                            Pt2dr YesHom(AllKps->at(2*n).at(Voisins.begin()->first).getPoint().x,AllKps->at(2*n).at(Voisins.begin()->first).getPoint().y);
                            ImAndPt ImPt={ThermalImages[n],YesHom};
                            Multiplicity.push_back(ImPt);
                        }
                        else
                        {
                           MaskFoundKps[2*n]=0;
                        }
                    }
                    else
                    {
                        MaskFoundKps[2*n]=0; // There is no found keypoint in this viewable image
                    }

                    Voisins.clear(); // Clear Voisins which contains the Nearest Neighbours to a point
                    AllTrees->at(2*n+1)->voisins(Ptslave, distMax, Voisins);
                    if (Voisins.size()>0)
                    {
                        //std::cout<<"============> First index of neighbours: "<<Voisins.begin()->first<<"    Relevant keypoints size:  "<<AllKps->at(2*n+1).size()<<endl;
                        if(WithinLimits(Voisins.begin()->first,AllKps->at(2*n+1)))
                        {
                            ImAndPt ImPt={VisualImages[n],Voisins.begin()->second};
                            Multiplicity.push_back(ImPt);
                        }
                        else
                        {
                            MaskFoundKps[2*n+1]=0;
                        }
                    }
                    else
                    {
                        MaskFoundKps[2*n+1]=0; // There is no found keypoint in this viewable image
                    }
                }
                else
                {
                    if (WhichIsSeen[2*n]) //Its homologous Thermal image
                    {
                        Orient_Image ImVslave(Oris_VIS_dir,VisualImages[n],aICNM);
                        Pt2dr Ptslave= ImVslave.getCam()->Ter2Capteur(pTerrain);
                        Voisins.clear(); // Clear Voisins which contains the Nearest Neighbours to a point
                        AllTrees->at(2*n)->voisins(Ptslave, distMax, Voisins);
                        if (Voisins.size()>0)
                        {
                            //std::cout<<"============> First index of neighbours: "<<Voisins.begin()->first<<"    Relevant keypoints size:  "<<AllKps->at(2*n).size()<<endl;
                            if (WithinLimits(Voisins.begin()->first, AllKps->at(2*n)))
                            {
                                Pt2dr YesHom(AllKps->at(2*n).at(Voisins.begin()->first).getPoint().x,AllKps->at(2*n).at(Voisins.begin()->first).getPoint().y);
                                ImAndPt ImPt={ThermalImages[n],YesHom};
                                Multiplicity.push_back(ImPt);
                            }
                            else
                            {
                                MaskFoundKps[2*n]=0;
                            }
                        }
                        else
                        {
                            MaskFoundKps[2*n]=0; // There is no found keypoint in this viewable image
                        }
                    }
                }
            }

            // check also if Multiplicity does invlove thermal and visual images and MaskFoundKps
            if (Multiplicity.size()>=4)
               {

                    // define a name to the Appui Point
                    string Namept="Pt_ImVis"+ ToString((int)i)+"_"+ToString((int)k);


                    /**************************************************************/
                    // store 3d point coordinates in cDicoAppuisFlottants
                    cOneAppuisDAF Point3D;
                    Point3D.Pt()=pTerrain;
                    Point3D.NamePt()=Namept;
                    Pt3dr Incertitude(1,1,1);
                    Point3D.Incertitude()=Incertitude;
                    All3Dpts.OneAppuisDAF().push_back(Point3D);
                    /**************************************************************/

                    // Fill cMesureAppuisFlottants relative to master image:: Here Visual
                    cOneMesureAF1I Measure;
                    Measure.NamePt()=Namept;
                    Measure.PtIm()=Pt2dr(AllKps->at(2*i+1).at(k).getPoint().x,AllKps->at(2*i+1).at(k).getPoint().y);
                    //std::cout<<"===========> Mesure 2d Master image: "<<Measure.PtIm()<<endl;
                    SetAppuis.at(2*i+1).OneMesureAF1I().push_back(Measure);
                    // Take a look at MaskFoundKps
                    uint iMul=0;
                    for (uint p=0;p<ThermalImages.size();p++)
                    {
                        if (MaskFoundKps[2*p]) // There is a keypoint in a thermal image
                        {
                            cOneMesureAF1I Measure;
                            Measure.NamePt()=Namept;
                            ELISE_ASSERT(ThermalImages[p]==Multiplicity.at(iMul).ImageName," Something went wrong in Multiplicity or conventions")
                            Measure.PtIm()=Multiplicity.at(iMul).MesureImage;
                            //std::cout<<" Index iMul:a thermal image: "<<iMul<<"   Measure 2d slave image: "<<Measure.PtIm()<<endl;
                            SetAppuis.at(2*p).OneMesureAF1I().push_back(Measure);
                            if(iMul<Multiplicity.size()-1){iMul++;}
                        }
                        if (MaskFoundKps[2*p+1])  // There is a KeyPoint in a visual image
                        {
                            cOneMesureAF1I Measure;
                            Measure.NamePt()=Namept;
                            ELISE_ASSERT(VisualImages[p]==Multiplicity.at(iMul).ImageName," Something went wrong in Multiplicity or conventions")
                            Measure.PtIm()=Multiplicity.at(iMul).MesureImage;
                            //std::cout<<" Index iMul:an optical image: "<<iMul<<"   Measure 2d slave image: "<<Measure.PtIm()<<endl;
                            SetAppuis.at(2*p+1).OneMesureAF1I().push_back(Measure);
                            if(iMul<Multiplicity.size()-1){iMul++;}
                        }
                    }
               }

        }

    }
    Depth.raz();
}



// Create all set of ground points and their corresponding 2D points in thermal images
int count=0;
   for (uint kk=0;kk<ThermalImages.size();kk++)
    {
       if(SetAppuis.at(2*kk).OneMesureAF1I().size())
         {
             AllSet.MesureAppuiFlottant1Im().push_back(SetAppuis.at(2*kk));
             count++;
         }
    }
   std::string Mesure2D="Mesure-2D.xml";
   std::string Mesure3D="Mesure-3D.xml";

   if (count)
   {
       MakeFileXML(AllSet,Mesure2D);
      // Write DicoAppuisFlottant into file Mesure-3D.xml
       MakeFileXML(All3Dpts,Mesure3D);
   }



// delete all trees AND Keypoint containers

for (uint i=0; i<AllTrees->size();i++)
{
    delete AllTrees->at(i);
}

delete AllTrees;

delete AllKps; delete AllKpsThermalHomog;

//=====================Computing thermo-optical tie points==========================================//



// IR/RGB registration: Homol folder contains all correspondences
std::string  HomolfileInter="./HomolInter";
ElPackHomologue HomologousPtsInter;

//Update Depth Map
Depth.raz();


if (!ELISE_fp::IsDirectory(HomolfileInter)) // Homolfile is not created
{
    std::cout<<"********************************************************************************************\n"<<
               "*********************************************************************************************\n"<<
               " Computing thermo-optical tie points using the homography predictor and the 3D optical model \n"<<
               "*********************************************************************************************\n"<<
               "**********************************************************************************************\n";

    ELISE_fp::MkDir(HomolfileInter); // create this directory

    for (uint i=0;i<ImCpls.size();i++)
    {
        //Clear all containers
        Kps1.clear();Kps1S.clear();Kps2.clear();Kps2S.clear();
        Kps1Skp.clear();Kps2Skp.clear();
        Kps1H.clear();Kps2H.clear();Kps1HS.clear();Kps2HS.clear();

        // Clear the set of homologous points
        HomologousPtsInter.clear();


        string ImageTh1=WhichThermalImage(ImCpls.at(i).N1(),ThermalImages);
        string ImageTh2=WhichThermalImage(ImCpls.at(i).N2(),ThermalImages);
        std::cout<<"===========>>> Possible Combinations \n";
        std::cout<<"  Couple ===========>  "<<ImCpls.at(i).N1()<<"        "<<ImageTh2<<endl;


        //Associated Pastis directories that cover the outlined combinations
        std::string DirPastisVis=HomolfileInter + "/Pastis" + ImCpls.at(i).N1();
        std::string DirPastisTh=HomolfileInter + "/Pastis" + ImageTh1;


        // Optical Pastis Dir
        if (!ELISE_fp::exist_file(DirPastisVis))
        {
            //create directory Pastis
            ELISE_fp::MkDir(DirPastisVis);
        }

        // Thermal Pastis Dir
        if (!ELISE_fp::exist_file(DirPastisTh))
        {
            //create directory Pastis
            ELISE_fp::MkDir(DirPastisTh);
        }

        // Files where matches are to be stored
        string file12VisTir=DirPastisVis + "/" + ImageTh2 + ".txt";
        string file12TirVis=DirPastisTh + "/" + ImCpls.at(i).N2() + ".txt";


        //First Files Vis=====>Tir
        if (!ELISE_fp::exist_file(file12VisTir))
        {


            //First optical image: get Keypoints ==> MSD
            /******************************************************/
            string filecpleIm1=Kpsfile + "/" + ImCpls.at(i).N1() + ".txt";
            Readkeypoints(Kps1,filecpleIm1);
            /*******************************************************/
            //First optical image: get Keypoints ==> SIFT
            /*******************************************************/
            string filecpleIm1S= KpsfileSIFT + "/" + ImCpls.at(i).N1() +".key";
            read_siftPoint_list(filecpleIm1S,Kps1S);
            /*******************************************************/
            // Second thermal image: get Keypoints ==> MSD
            /*******************************************************/
            string filecpleIm2=Kpsfile + "/"+ImageTh2 + ".txt";
            Readkeypoints(Kps2,filecpleIm2);
            /*******************************************************/
            // Second thermal image: get Keypoints ==> SIFT
            /*******************************************************/
           string filecpleIm2S= KpsfileSIFT + "/" + ImageTh2+ ".key";
           read_siftPoint_list(filecpleIm2S,Kps2S);
           /*******************************************************/
            // Apply the computed homography to the thermal images ImageTh2

           /****************MSD MSD MSD MSD MSD ******************/
            Kps2H=NewSetKpAfterHomog(Kps2,Hout);
            /***************SIFT SIFT SIFT SIFT SIFT *************/
            Kps2HS=NewSetKpAfterHomog(Kps2S,Hout);


            std::cout<<"===========> MSD Points  "<<Kps1.size()<<"    ||   "<<Kps2H.size()<<endl;
            std::cout<<"===========> SIFT Points "<<Kps1S.size()<<"   ||   "<<Kps2HS.size()<<endl;


            //At this step, we can merge SIFT and MSD keypoints

            Kps1Skp= FromSiftP2KeyP(Kps1S); // move to KeyPoint frame
            Kps2Skp= FromSiftP2KeyP(Kps2S); // move to KeyPoint frame

            Kps1.insert(Kps1.end(),Kps1Skp.begin(),Kps1Skp.end());  //optical keypoints
            Kps2.insert(Kps2.end(),Kps2Skp.begin(),Kps2Skp.end());  //thermal keypoints

            Kps2H.insert(Kps2H.end(),Kps2HS.begin(),Kps2HS.end());  // Homography-applied thermal keypoints




            //Use the oriented Visual Images to compute image1==>Terrain=>image2
            Orient_Image ImV1(Oris_VIS_dir,ImCpls.at(i).N1(),aICNM);
            Orient_Image ImV2(Oris_VIS_dir,ImCpls.at(i).N2(),aICNM);


            //Apply Image1 ==> Terrain operation having into consideration a depth image
            // from ZBufferRaster directory
            string FileDepthImV1="./Tmp-ZBuffer/" + PlyFileIn + "/" + ImCpls.at(i).N1() + "/" + ImCpls.at(i).N1() + "_ZBuffer_DeZoom1.tif";
            ELISE_ASSERT(DoesFileExist(FileDepthImV1.c_str()),"The Depth image relative to visual Image is not found");

            // load Depth image corresponding to the visual image needed for the task
            Depth=Im2D<REAL4,REAL>::FromFileBasic(FileDepthImV1);


            //Creating and filling the KDTree structure
            /*********************************************************************/
            ArbreKD * SlaveTree= new ArbreKD(Pt_of_Point, box, Kps2H.size(), 1.0);
            for (uint i=0; i<Kps2H.size(); i++) {
                SlaveTree->insert(pair<int,Pt2dr>(i, Kps2H.at(i)));
            }
            /*********************************************************************/

            for (uint j=0;j<Kps1.size();j++)
            {
                // Sift repetetive points are caused by scale space definition, onr should filter theses occurences
                // so that one among them is only processed
                if (j>0 && (Kps1.at(j).getPoint().x==Kps1.at(j-1).getPoint().x && Kps1.at(j).getPoint().y==Kps1.at(j-1).getPoint().y))
                {
                    continue;
                }

                double Prof;
                // Added Condition due to an error of segmentation: point where depth is not defined
                /*************************************************************************************/
                Pt2dr Pdr(Kps1.at(j).getPoint().x,Kps1.at(j).getPoint().y);
                if (IncludedInDepthIm(Pdr,Depth.sz()))
                {
                    Prof=Depth.BilinIm()->Get(Pdr);
                }
                else
                {
                    Prof=-1;
                }
                /**************************************************************************************/


                if (Prof!=-1)
                {

                    Pt3dr pTerrain= ImV1.getCam()->ImDirEtProf2Terrain(Pdr,Prof,ImV1.getCam()->DirVisee());
                    Pt2dr PtImage2= ImV2.getCam()->Ter2Capteur(pTerrain);

                    Voisins.clear();
                    SlaveTree->voisins(PtImage2, distMax, Voisins);

                    if (Voisins.size()>0)
                     {
                        Pt2dr Pnew(Kps2.at(Voisins.begin()->first).getPoint().x,Kps2.at(Voisins.begin()->first).getPoint().y);
                        HomologousPtsInter.Cple_Add(ElCplePtsHomologues(Pdr,Pnew));
                     }
                }
            }
            delete SlaveTree;
            Depth.raz();
            HomologousPtsInter.StdPutInFile(file12VisTir);
        }

       //A new combination of images
        HomologousPtsInter.clear();
        Depth.raz();

        //Clear all containers
        Kps1.clear();Kps1S.clear();Kps2.clear();Kps2S.clear();
        Kps1Skp.clear();Kps2Skp.clear();
        Kps1H.clear();Kps2H.clear();Kps1HS.clear();Kps2HS.clear();




        //Now the second way Tir ====> Vis
        std::cout<<"  Couple ===========>  "<<ImageTh1<<"        "<<ImCpls.at(i).N2()<<endl;

        if (!ELISE_fp::exist_file(file12TirVis))
        {


            //First thermal image: get Keypoints ==> MSD
            /******************************************************/
            string filecpleIm1=Kpsfile + "/" + ImageTh1 + ".txt";
            Readkeypoints(Kps1,filecpleIm1);
            /*******************************************************/
            //First thermal image: get Keypoints ==> SIFT
            /*******************************************************/
            string filecpleIm1S= KpsfileSIFT + "/" + ImageTh1 +".key";
            read_siftPoint_list(filecpleIm1S,Kps1S);
            /*******************************************************/
            // Second optical image: get Keypoints ==> MSD
            /*******************************************************/
            string filecpleIm2=Kpsfile + "/"+ImCpls.at(i).N2() + ".txt";
            Readkeypoints(Kps2,filecpleIm2);
            /*******************************************************/
            // Second optical image: get Keypoints ==> SIFT
            /*******************************************************/
           string filecpleIm2S= KpsfileSIFT + "/" + ImCpls.at(i).N2()+ ".key";
           read_siftPoint_list(filecpleIm2S,Kps2S);
           /*******************************************************/
            // Apply the computed homography to the thermal images ImageTh1: Now Master image

           /****************MSD MSD MSD MSD MSD ******************/
            Kps1H=NewSetKpAfterHomog(Kps1,Hout);
            /***************SIFT SIFT SIFT SIFT SIFT *************/
            Kps1HS=NewSetKpAfterHomog(Kps1S,Hout);


            std::cout<<"===========> MSD Points  "<<Kps1.size()<<"    ||  "<<Kps2.size()<<endl;
            std::cout<<"===========> SIFT Points "<<Kps1S.size()<<"   ||  "<<Kps2S.size()<<endl;


            //At this step, we can merge SIFT and MSD keypoints

            Kps1Skp= FromSiftP2KeyP(Kps1S); // move to KeyPoint frame
            Kps2Skp= FromSiftP2KeyP(Kps2S); // move to KeyPoint frame

            Kps1.insert(Kps1.end(),Kps1Skp.begin(),Kps1Skp.end());  //Thermal keypoints
            Kps2.insert(Kps2.end(),Kps2Skp.begin(),Kps2Skp.end());  //Optical keypoints

            Kps1H.insert(Kps1H.end(),Kps1HS.begin(),Kps1HS.end());  // Homography-applied thermal keypoints




            //Use the oriented Visual Images to compute image1==>Terrain=>image2
            Orient_Image ImV1(Oris_VIS_dir,ImCpls.at(i).N1(),aICNM);
            Orient_Image ImV2(Oris_VIS_dir,ImCpls.at(i).N2(),aICNM);


            //Apply Image1 ==> Terrain operation having into consideration a depth image
            // from ZBufferRaster directory
            string FileDepthImV1="./Tmp-ZBuffer/" + PlyFileIn + "/" + ImCpls.at(i).N1() + "/" + ImCpls.at(i).N1() + "_ZBuffer_DeZoom1.tif";
            ELISE_ASSERT(DoesFileExist(FileDepthImV1.c_str()),"The Depth image relative to visual Image is not found");

            // load Depth image corresponding to the visual image needed for the task
            Depth=Im2D<REAL4,REAL>::FromFileBasic(FileDepthImV1);

            //Creating and filling the KDTree structure
            /*********************************************************************/
            ArbreKD * SlaveTree= new ArbreKD(Pt_of_Point, box, Kps2.size(), 1.0);
            for (uint i=0; i<Kps2.size(); i++) {
                Pt2dr P2d(Kps2.at(i).getPoint().x,Kps2.at(i).getPoint().y);
                SlaveTree->insert(pair<int,Pt2dr>(i, P2d));
            }
            /*********************************************************************/

            for (uint j=0;j<Kps1H.size();j++)
            {
                // Sift repetetive points are caused by scale space definition, onr should filter theses occurences
                // so that one among them is only processed
                if (j>0 && (Kps1H.at(j).x==Kps1H.at(j-1).x && Kps1H.at(j).y==Kps1H.at(j-1).y))
                {
                    continue;
                }

                double Prof;
                // Added Condition due to an error of segmantation: point where depth is not defined
                /*************************************************************************************/
                if (IncludedInDepthIm(Kps1H.at(j),Depth.sz()))
                {
                    Prof=Depth.BilinIm()->Get(Kps1H.at(j));
                }
                else
                {
                    Prof=-1;
                }
                /**************************************************************************************/


                if (Prof!=-1)
                {

                    Pt3dr pTerrain= ImV1.getCam()->ImDirEtProf2Terrain(Kps1H.at(j),Prof,ImV1.getCam()->DirVisee());
                    Pt2dr PtImage2= ImV2.getCam()->Ter2Capteur(pTerrain);

                    Voisins.clear();
                    SlaveTree->voisins(PtImage2, distMax, Voisins);

                    if (Voisins.size()>0)
                     {
                        Pt2dr P1(Kps1.at(j).getPoint().x, Kps1.at(j).getPoint().y);
                        Pt2dr Pnew(Kps2.at(Voisins.begin()->first).getPoint().x,Kps2.at(Voisins.begin()->first).getPoint().y);
                        HomologousPtsInter.Cple_Add(ElCplePtsHomologues(P1,Pnew));
                     }
                }
            }
            delete SlaveTree;
            Depth.raz();
            HomologousPtsInter.StdPutInFile(file12TirVis);
        }

    }

    //Update Depth image
    Depth.raz();
/********************************************************************/
// Couples of homologous images (THERMAL AND OPTICAL)
   for (uint i=0; i<ThermalImages.size();i++)
   {

       // Thermal image ======> Its corresponding optical image

              //Clear all containers
               Kps1.clear();Kps1S.clear();Kps2.clear();Kps2S.clear();
               Kps1Skp.clear();Kps2Skp.clear();
               Kps1H.clear();Kps2H.clear();Kps1HS.clear();Kps2HS.clear();

               // Clear the set of homologous points
               HomologousPtsInter.clear();
               //std::cout<<"===========>> Combination of homologous images  \n";
               std::cout<<"  Couple ===========>  "<<ThermalImages[i]<<"        "<<VisualImages[i]<<endl;


               //Associated Pastis directories that cover the outlined combinations
               std::string DirPastisc1=HomolfileInter + "/Pastis" + ThermalImages[i];

               // Files where matches are to be stored
               string file12c1= DirPastisc1 + "/" + VisualImages[i] + ".txt";


               //First Files Vis=====>Tir
               if (!ELISE_fp::exist_file(file12c1))
               {


                   //First thermal image: get Keypoints ==> MSD
                   /******************************************************/
                   string filecpleIm1=Kpsfile + "/" + ThermalImages[i] + ".txt";
                   Readkeypoints(Kps1,filecpleIm1);
                   /*******************************************************/
                   //First optical image: get Keypoints ==> SIFT
                   /*******************************************************/
                   string filecpleIm1S= KpsfileSIFT + "/" + ThermalImages[i] +".key";
                   read_siftPoint_list(filecpleIm1S,Kps1S);
                   /*******************************************************/
                   // Second thermal image: get Keypoints ==> MSD
                   /*******************************************************/
                   string filecpleIm2=Kpsfile + "/"+VisualImages[i] + ".txt";
                   Readkeypoints(Kps2,filecpleIm2);
                   /*******************************************************/
                   // Second thermal image: get Keypoints ==> SIFT
                   /*******************************************************/
                  string filecpleIm2S= KpsfileSIFT + "/" + VisualImages[i]+ ".key";
                  read_siftPoint_list(filecpleIm2S,Kps2S);
                  /*******************************************************/
                   // Apply the computed homography to the thermal images Master image

                  /****************MSD MSD MSD MSD MSD ******************/
                   Kps1H=NewSetKpAfterHomog(Kps1,Hout);
                   /***************SIFT SIFT SIFT SIFT SIFT *************/
                   Kps1HS=NewSetKpAfterHomog(Kps1S,Hout);


                   std::cout<<"===========> MSD Points  "<<Kps1H.size()<<"   ||   "<<Kps2.size()<<endl;
                   std::cout<<"===========> SIFT Points "<<Kps1S.size()<<"   ||   "<<Kps2S.size()<<endl;


                   //At this step, we can merge SIFT and MSD keypoints

                   Kps1Skp= FromSiftP2KeyP(Kps1S); // move to KeyPoint frame
                   Kps2Skp= FromSiftP2KeyP(Kps2S); // move to KeyPoint frame

                   Kps1.insert(Kps1.end(),Kps1Skp.begin(),Kps1Skp.end());  //optical keypoints
                   Kps2.insert(Kps2.end(),Kps2Skp.begin(),Kps2Skp.end());  //thermal keypoints

                   Kps1H.insert(Kps1H.end(),Kps1HS.begin(),Kps1HS.end());  // Homography-applied thermal keypoints

                   //Creating and filling the KDTree structure
                   /*********************************************************************/
                   ArbreKD * SlaveTree= new ArbreKD(Pt_of_Point, box, Kps2.size(), 1.0);
                   for (uint i=0; i<Kps2.size(); i++) {
                       Pt2dr Pdr(Kps2.at(i).getPoint().x,Kps2.at(i).getPoint().y);
                       SlaveTree->insert(pair<int,Pt2dr>(i, Pdr));
                   }
                   /*********************************************************************/

                   for (uint j=0;j<Kps1H.size();j++)
                   {
                       // Sift repetetive points are caused by scale space definition, onr should filter theses occurences
                       // so that one among them is only processed
                       if (j>0 && (Kps1H.at(j).x==Kps1H.at(j-1).x && Kps1H.at(j).y==Kps1H.at(j-1).y))
                       {
                           continue;
                       }
                       Voisins.clear();
                       SlaveTree->voisins(Kps1H.at(j), rayMax, Voisins);

                       if (Voisins.size()>0)
                        {
                           Pt2dr P1(Kps1.at(j).getPoint().x,Kps1.at(j).getPoint().y);
                           Pt2dr Pnew(Kps2.at(Voisins.begin()->first).getPoint().x,Kps2.at(Voisins.begin()->first).getPoint().y);
                           HomologousPtsInter.Cple_Add(ElCplePtsHomologues(P1,Pnew));
                        }
                   }
                   delete SlaveTree;
                   HomologousPtsInter.StdPutInFile(file12c1);
               }
     // Optical images =========> Thermal image
     std::cout<<"  Couple ===========>  "<<VisualImages[i]<<"        "<<ThermalImages[i]<<endl;


     //Clear all containers
      Kps1.clear();Kps1S.clear();Kps2.clear();Kps2S.clear();
      Kps1Skp.clear();Kps2Skp.clear();
      Kps1H.clear();Kps2H.clear();Kps1HS.clear();Kps2HS.clear();

      // Clear the set of homologous points
      HomologousPtsInter.clear();


      //Associated Pastis directories that cover the outlined combinations
      std::string DirPastisc2=HomolfileInter + "/Pastis" + VisualImages[i];

      // Files where matches are to be stored
      string file12c2= DirPastisc2 + "/" + ThermalImages[i] + ".txt";


      //First Files Vis=====>Tir
      if (!ELISE_fp::exist_file(file12c2))
      {


          //First thermal image: get Keypoints ==> MSD
          /******************************************************/
          string filecpleIm1=Kpsfile + "/" + VisualImages[i] + ".txt";
          Readkeypoints(Kps1,filecpleIm1);
          /*******************************************************/
          //First optical image: get Keypoints ==> SIFT
          /*******************************************************/
          string filecpleIm1S= KpsfileSIFT + "/" + VisualImages[i] +".key";
          read_siftPoint_list(filecpleIm1S,Kps1S);
          /*******************************************************/
          // Second thermal image: get Keypoints ==> MSD
          /*******************************************************/
          string filecpleIm2=Kpsfile + "/"+ThermalImages[i] + ".txt";
          Readkeypoints(Kps2,filecpleIm2);
          /*******************************************************/
          // Second thermal image: get Keypoints ==> SIFT
          /*******************************************************/
         string filecpleIm2S= KpsfileSIFT + "/" + ThermalImages[i]+ ".key";
         read_siftPoint_list(filecpleIm2S,Kps2S);
         /*******************************************************/
          // Apply the computed homography to the thermal images Master image

         /****************MSD MSD MSD MSD MSD ******************/
          Kps2H=NewSetKpAfterHomog(Kps2,Hout);
          /***************SIFT SIFT SIFT SIFT SIFT *************/
          Kps2HS=NewSetKpAfterHomog(Kps2S,Hout);


          std::cout<<"===========> MSD Points  "<<Kps1.size()<<"    ||   "<<Kps2H.size()<<endl;
          std::cout<<"===========> SIFT Points "<<Kps1S.size()<<"   ||   "<<Kps2S.size()<<endl;


          //At this step, we can merge SIFT and MSD keypoints

          Kps1Skp= FromSiftP2KeyP(Kps1S); // move to KeyPoint frame
          Kps2Skp= FromSiftP2KeyP(Kps2S); // move to KeyPoint frame

          Kps1.insert(Kps1.end(),Kps1Skp.begin(),Kps1Skp.end());  //optical keypoints
          Kps2.insert(Kps2.end(),Kps2Skp.begin(),Kps2Skp.end());  //thermal keypoints

          Kps2H.insert(Kps2H.end(),Kps2HS.begin(),Kps2HS.end());  // Homography-applied thermal keypoints

          //Creating and filling the KDTree structure
          /*********************************************************************/
          ArbreKD * SlaveTree= new ArbreKD(Pt_of_Point, box, Kps2H.size(), 1.0);
          for (uint i=0; i<Kps2H.size(); i++) {
              SlaveTree->insert(pair<int,Pt2dr>(i, Kps2H.at(i)));
          }
          /*********************************************************************/

          for (uint j=0;j<Kps1.size();j++)
          {
              // Sift repetetive points are caused by scale space definition, onr should filter theses occurences
              // so that one among them is only processed
              if (j>0 && (Kps1.at(j).getPoint().x==Kps1.at(j-1).getPoint().x && Kps1.at(j).getPoint().y==Kps1.at(j-1).getPoint().y))
              {
                  continue;
              }
              Voisins.clear();
              Pt2dr P1(Kps1.at(j).getPoint().x,Kps1.at(j).getPoint().y);
              SlaveTree->voisins(P1, rayMax, Voisins);

              if (Voisins.size()>0)
               {

                  Pt2dr Pnew(Kps2.at(Voisins.begin()->first).getPoint().x,Kps2.at(Voisins.begin()->first).getPoint().y);
                  HomologousPtsInter.Cple_Add(ElCplePtsHomologues(P1,Pnew));
               }
          }
          delete SlaveTree;
          HomologousPtsInter.StdPutInFile(file12c2);
      }
   }
}
else
{
    std::cout<<"********************************************************************************************\n"<<
               "*********************************************************************************************\n"<<
               " Thermo-optical points are already computed and there is a filled ./HomolInter directory      \n"<<
               "*********************************************************************************************\n"<<
               "**********************************************************************************************\n";
}




/*
// create binary masks


    for (uint i=0; i<aSetImTest.size();i++)
    {
        Tiff_Im Mask=Tiff_Im::UnivConvStd(aSetImTest.at(i));
        Im2D<U_INT1,INT> Maskk=Im2D<U_INT1,INT>(Mask.sz().x,Mask.sz().y);

        ELISE_COPY
              (
                  Maskk.all_pts(),
                  Mask.in(),
                  Maskk.out()
              );


        Im2D_Bits<1> ImagedeMask=Im2D_Bits<1>(Mask.sz().x,Mask.sz().y);

        for (int x=0; x<Mask.sz().x; x++)
        {
            for (int y=0; y<Mask.sz().y; y++)
            {
                Pt2di point(x,y);
                int val=Maskk.GetI(point);
                if (val>0)
                {
                    ImagedeMask.SetI(point, 1);
                }
                else
                {
                    ImagedeMask.SetI(point, 0);
                }
            }
        }
        std::string nameMask="Mask_Bin"+aSetImTest.at(i);

        ELISE_COPY
          (
              ImagedeMask.all_pts(),
              ImagedeMask.in(),
              Tiff_Im(
                  nameMask.c_str(),
                  ImagedeMask.sz(),
                  GenIm::bits1_msbf,
                  Tiff_Im::No_Compr,
                  Tiff_Im::BlackIsZero,
                  Tiff_Im::Empty_ARG ).out()
          );
    }*/

return EXIT_SUCCESS ;
}
