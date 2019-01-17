#include "cmsdappli.h"

/*

*/

cMSD1Im::cMSD1Im(int argc,char ** argv):
    mTmpDir("Tmp-MM-Dir/"),
    mDebug(1),
    msd(),
    mTh(0.02),// a posteriori filter: low value
    mPR(3),
    mSAR(5),
    mKNN(5),
    mNMS(5),// a posteriori filter: low value
    mSc(-1)
{
    // warn, when PA 3 SAR 3 and NMS 5, bug --> should add warnings

    std::string aBidon;
    // provide me some pt with descriptor on saliency and Ratio=0.8, orientation size mSize/2 on row im and descriptor mSize/2, on saliency DZ0 map. mainly DZ0 pt. Still lot of outliers. I try other localscale param for orientate and describe, nothing betters than that
    //double aTh(0.001);
    //int aPR(3),aSAR(5),aKNN(5),aNMS(3);
    // PR 5 SAR 3 NMS 3 KNN 5 Ratio=0.8, orientation size mSize/2 on row im and descriptor mSize/2, on saliency DZ0 map. mainly NOT DZ0 pt. not that many outlier
    // PR 5 SAR 3 NMS 3 KNN 5 Ratio=0.8, orientation size m_patch_radisu on row im DZx and descriptor m_patch_radius, on saliency DZx map--> do not work! should be equivalent than line up!! wtf?? but right that saliency map change rapidly with the scale

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mNameIm,"input",eSAM_IsExistFile)
                    << EAMC(aBidon,"-o")
                    << EAMC(mOut,"output"),
        LArgMain()  << EAM(mDebug,"Debug",true,"Debug mode, def false")
                    << EAM(mPR,"PR",true,"patch radius, def 3.")
                    << EAM(mSAR,"SAR",true,"search area radius, def 5.")
                    << EAM(mTh,"Th",true,"Threshold of saliency, def 0.01, ok with SFS filter")
                    << EAM(mNMS,"NMS",true,"Non-Maxima Suppression (on saliency map) radius, def 3.")
                    << EAM(mKNN,"KNN",true,"KNN neighbour, def 5.")
                    << EAM(mSc,"NbSc",true,"number of scale, def -1.")
                    << EAM(mTmpDir,"Dir",true,"Directory used in debug mode to store intermediate results, def Tmp-MM-Dir")
                );

    mICNM = cInterfChantierNameManipulateur::BasicAlloc("./");


     // handling of both thermic optris image (1 cannal, 16 bits) and camlight images (1 cannal, 8 bits)
     Tiff_Im Im=Tiff_Im::UnivConvStd(mNameIm);
     if (Im.type_el()==GenIm::u_int2) {
         mIm=Im2D_U_INT1(Im.sz().x,Im.sz().y,0);
         double aZMin,aZMax;
         ELISE_COPY
         (
             Im.all_pts(),
             Im.in(),
             VMax(aZMax)|VMin(aZMin)
         );

         ELISE_COPY(Im.all_pts(),255.0*(Im.in()-aZMin)/(aZMax-aZMin),mIm.out());

     } else { mIm=Im2D_U_INT1::FromFileStd(mNameIm);}


     initMSD();
     msd.detect(mIm);
     msd.orientationAndDescriptor();
     msd.writeKp(mOut);
     if (mDebug) std::cout << "For image " << mNameIm << ", I have found " << msd.Kps().size() << " MSD points \n";

}


void cMSD1Im::MSDBanniere()
{
    std::cout <<  "\n";
    std::cout <<  " **************************\n";
    std::cout <<  " *     M-aximum           *\n";
    std::cout <<  " *     S-elf              *\n";
    std::cout <<  " *     D-issimilarity     *\n";
    std::cout <<  " **************************\n";
    std::cout <<  " Tuned to work with SFS radiometric filter (@SFS)\n";
    std::cout <<  " Patch Radius: " << msd.getPatchRadius() << ", Search area radius " <<  msd.getSearchAreaRadius() << " Saliency thresh " <<  msd.getThSaliency()      <<"\n";
    std::cout <<  " Non-Maxima Suppression radius : " << msd.getNMSRadius() << ", nb KNN " <<  msd.getKNN() << " \n\n";
}

//===============================================================================//
// Call for wallis filter: locally equalized image
//===============================================================================//

void wallis( Im2D<U_INT1,INT> &image, Im2D<U_INT1,INT> &WallEqIm)
{

    // image should be changed to gray space

    int n = image.sz().x;
    int m = image.sz().y;
    // Block dimension in i and j
    int dim_n = 40, dim_m = 40;
    int N_Block = n/dim_n;
    //cout<<"N_Block\t"<<N_Block<<endl;
    int M_Block = m/dim_m;
    //cout<<"M_Block\t"<<M_Block<<endl;
    int resto_n = n%dim_n;
    int resto_m = m%dim_m;
    int *dimension_x = new int[N_Block];
    int *dimension_y = new int[M_Block];

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


    //cout <<"Wallis filter: window size of " << dim_n << "x" << dim_m << "pix, desired mean DN of " << mf <<", desired std of " << sf<< endl;

    int px = 0, py = 0;

    Im2D<REAL4,REAL8> Coeff_R0 = Im2D<REAL4,REAL8>(N_Block, M_Block);
    Im2D<REAL4,REAL8> Coeff_R1 = Im2D<REAL4,REAL8>(N_Block, M_Block);

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

            double aSP,aSomZ,aSomZ2;

            ELISE_COPY
            (
                rectangle(aP0,aP0+aSz),
                Virgule(1,aTF,Square(aTF)),
                Virgule
                (
                     sigma(aSP),// number of obs
                     sigma(aSomZ),// sum of DN value
                     sigma(aSomZ2)// sum of squared DN
                )
            );

            aSomZ /= aSP; // mean value
            aSomZ2 /= aSP; // sum of squared DN/ number of DN
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
                (Imagefloat.in()-Minvalue)*255.0/(Maxvalue-Minvalue),// to protect from overfloat issue
                WallEqIm.out()
               );
    delete [] dimension_x;
    delete [] dimension_y;
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
        // not an RGB image
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

//===============================================================================//
/*      Routine that processes an image: Im2D--> Lab and Wallis filter            */
//===============================================================================//
// i should check that it is usefull somehow
void Migrate2Lab2wallis(Im2D<U_INT1,INT> &image, Im2D<U_INT1,INT> & Output)
{
    RGB2Lab_b rgbtolab=RGB2Lab_b(3,0,0,0,true);
    Pt2di mImgSz(image.sz().x,image.sz().y);

    Im2D<U_INT1,INT>* mImgR(0);
    Im2D<U_INT1,INT>* mImgG(0);
    Im2D<U_INT1,INT>* mImgB(0);

    mImgR=new Im2D<U_INT1,INT>(mImgSz.x,mImgSz.y);
    mImgG=new Im2D<U_INT1,INT>(mImgSz.x,mImgSz.y);
    mImgB=new Im2D<U_INT1,INT>(mImgSz.x,mImgSz.y);
    ELISE_COPY(mImgR->all_pts(),image.in(),mImgR->out());
    ELISE_COPY(mImgG->all_pts(),image.in(),mImgG->out());
    ELISE_COPY(mImgB->all_pts(),image.in(),mImgB->out());

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

int MSD_main( int argc, char **argv)
{ 
    cMSD1Im appli(argc,argv);
    appli.MSDBanniere();

    return EXIT_SUCCESS;
}
