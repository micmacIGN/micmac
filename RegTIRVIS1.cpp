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
    cElRegex rgx("VIS-(.*).tif",10);  // need to change according to pattern
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
    cElRegex rgx("TIR-(.*).tif",10);
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
            LArgMain() //<< EAMC(TestDataDIRpat ,"Set of images to be used to estimate the homography",eSAM_IsPatFile)
                       << EAMC(Image_Pattern," Images pattern : Dir+ Name: prefixes Thermal: TIR, Visual: VIS", eSAM_IsPatFile)
                       //<< EAMC(Oris_VIS_dir," Orientation files for visual images",eSAM_IsExistDirOri)
                       //<< EAMC(PlyFileIn, " Ply file that will be used as a 3d model",eSAM_IsExistFile)
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
    SplitDirAndFile(aDirtest,aPatImTst,Image_Pattern);


    std::cout<<"==============> Test data directory: "<<aDirtest<<endl;
    std::cout<<"==============> Pattern image test: "<<aPatImTst<<endl;


    // Chantier Manipulateur
    cInterfChantierNameManipulateur * aICNM0=cInterfChantierNameManipulateur::BasicAlloc(aDirtest);
    const std::vector<std::string> aSetImTest = *(aICNM0->Get(aPatImTst));
    std::cout<<"==============> Param Chantier manipulateur set \n";


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
