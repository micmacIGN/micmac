#include "cthermicto8bits.h"

// color palette for RGB diplay of thermal images
int Deg2R_pal1(double aDeg , Pt2dr aRange=Pt2dr(23.0,50.0)){
    int aRes(0);
    double thresh1(aRange.x+14*(aRange.y-aRange.x)/27), thresh2(aRange.x+18*(aRange.y-aRange.x)/27);
    if (aDeg<=thresh1) aRes=0;
    if (aDeg>thresh1 && aDeg <=thresh2) aRes=(aDeg-thresh1)*(255.0/(thresh2-thresh1));
    if (aDeg>thresh2) aRes=255;
    //std::cout << " aDeg: " << aDeg << " , range " << aRange << " , thresh1 " << thresh1 << " and thresh2 " << thresh2 << ", result = " << aRes << "\n";
    return aRes;
}
int Deg2G_pal1(double aDeg, Pt2dr aRange=Pt2dr(23.0,50.0)){
    int aRes(0);
    double thresh1(aRange.x+1*(aRange.y-aRange.x)/27), thresh2(aRange.x+9*(aRange.y-aRange.x)/27), thresh3(aRange.x+14*(aRange.y-aRange.x)/27), thresh4(aRange.x+18*(aRange.y-aRange.x)/27);
    if (aDeg<=thresh1) return 0;
    if (aDeg>thresh1 && aDeg <=thresh2) aRes= (aDeg-thresh1)*(100/(thresh2-thresh1));
    if (aDeg>thresh2 && aDeg <=thresh3) aRes= 100+(aDeg-thresh2)*(155/(thresh3-thresh2));
    if (aDeg>thresh3 && aDeg <=thresh4) aRes= 255;
    if (aDeg>thresh4 && aDeg <=aRange.y) aRes= 255-(aDeg-thresh4)*(255/(aRange.y-thresh4));
    if (aDeg>aRange.y) aRes= 0;
    return aRes;
}
int Deg2B_pal1(double aDeg, Pt2dr aRange=Pt2dr(23.0,50.0)){
    int aRes(0);
    double thresh1(aRange.x+1*(aRange.y-aRange.x)/27), thresh2(aRange.x+8*(aRange.y-aRange.x)/27), thresh3(aRange.x+15*(aRange.y-aRange.x)/27);
    if (aDeg<=thresh1) aRes= 0;
    if (aDeg>thresh1 && aDeg <=thresh2) aRes= (aDeg-thresh1)*(255/(thresh2-thresh1));
    if (aDeg>thresh2 && aDeg <=thresh3) aRes= 255-(aDeg-thresh2)*(155/(thresh3-thresh2));
    if (aDeg>thresh3) aRes= 0;
    return aRes;
}

// conversion of Digital Number raw data thermal frame to degree and the other way around
double DN2Deg_Optris(int aVal){
    return ((double)aVal-1000.0)/10.0;
}
int Deg2DN_Optris(double aVal){
    return aVal*10+1000;
}
double DN2Deg_Vario(int aVal){
    return (double)aVal/100.0-273.15;
}
int Deg2DN_Vario(double aVal){
    return 100*(273.15+aVal);
}


cDeg2RGB::cDeg2RGB(double aDeg, int aR, int aG, int aB):mDeg(aDeg),mR(aR),mG(aG),mB(aB){}
void cMeasurePalDeg2RGB::saveMes(string aFileName){
    FILE * aFOut = FopenNN(aFileName.c_str(),"w","out");
    for (auto & mes : mMes){
        fprintf(aFOut,"%f %i %i %i\n",mes->deg(),mes->r(),mes->g(),mes->b());
    }
}


cMeasurePalDeg2RGB::cMeasurePalDeg2RGB(int argc,char ** argv):
    mDebug(0),
    mVario(1),
    mOptris(0),
    mNbCarToRemove(4), // MPD reorder , warning ...
    mPre(""),   // MPD reorder , warning ...
    mSu(""),
    mExt("jpg")
{

    std::string aImThermPat,aImJPGPat;
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(mDirT,"Directory for thermal images", eSAM_IsDir)
                << EAMC(aImThermPat,"Pattern for thermal images", eSAM_IsPatFile)
                << EAMC(mDirJPG,"Directory for JPG images", eSAM_IsDir)
                << EAMC(aImJPGPat,"Pattern for JPG images from thermal images", eSAM_IsPatFile)
                << EAMC(mOut,"txt File to save observation degree - RGB .",eSAM_IsExistFile )
                ,
                LArgMain()
                << EAM(mDebug,"Debug",true, "Print message in terminal to help debugging." )
                << EAM(mVario,"Vario",true, "Use formula to compute Deg from DN for VarioCam camera, def true." )
                << EAM(mOptris,"Optris",true, "Use formula to compute Deg from DN for optris camera, def false." )
                << EAM(mPre,"Prefix",true, "Prefix to go from thermal to RGB name, def none." )
                << EAM(mSu,"Sufix",true, "Sufix to go from thermal to RGB name, def none." )
                << EAM(mExt,"Ext",true, "extension of jpg images, def 'jpg'" )
                << EAM(mNbCarToRemove,"NbChar2Rem",true, "Number of character of prefix of thermal image to remove to get the associated JPG name, def = 4 (TIR_)" )
                );
    if (!MMVisualMode)
    {
        mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDirJPG);
        mLImJPG = mICNM->StdGetListOfFile(aImJPGPat);
        mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDirT);
        mLImTherm = mICNM->StdGetListOfFile(aImThermPat);

        for (auto & therm : mLImTherm ){
            std::string NameJPG=AssocTherm2JPG(therm);
            if (mDebug) std::cout << "Image thermique " << mDirT+therm << " will be compared to RGB " << mDirJPG+NameJPG << "\n";
            std::list<std::string>::iterator it;
            it = find (mLImJPG.begin(), mLImJPG.end(), NameJPG);
            if (it != mLImJPG.end()){
                // open thermal image and JPG, read 1 points per tile
                Im2D_U_INT2 imTh=Im2D_U_INT2::FromFileStd(mDirT+therm);
                std::string aName(mDirJPG+NameJPG);
                cISR_ColorImg  imRGB(aName);
                int Tile(100);
                for (int u(1); u*Tile<imTh.sz().x; u++){
                    for (int v(1); v*Tile<imTh.sz().y; v++){
                        Pt2di pt(u*Tile,v*Tile);
                        cISR_Color col=imRGB.get(pt);
                        double deg(0);
                        if (mVario) deg=DN2Deg_Vario(imTh.GetR(pt));
                        if (mOptris) deg=DN2Deg_Optris(imTh.GetR(pt));
                        mMes.push_back(new cDeg2RGB(deg, col.r(), col.g(),col.b()));
                    }
                }
            } else {
            if (mDebug) std::cout << "Image RGB " << mDirJPG+NameJPG << " do not exist or is not in the provided pattern \n";
            }
        }
        saveMes(mOut);
    }
}

std::string cMeasurePalDeg2RGB::AssocTherm2JPG(std::string aName){
 return mPre+ aName.substr(mNbCarToRemove, aName.size()-(3+mNbCarToRemove)) + mSu+mExt;
}





cThermicTo8Bits::cThermicTo8Bits(int argc,char ** argv) :
    mFullDir	("img.*.tif"),
    mPrefix ("8bits_"),
    mOverwrite (false),
    mOptris(1),
    mVario(0),
    mRGB(0),
    mDebug(0)
{
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(mFullDir,"thermal image pattern", eSAM_IsPatFile)
                << EAMC(mRangeT,"Temperature range (celcius degree), 8 bits output images will be stretched on this range"),
                LArgMain()  << EAM(mOverwrite,"F",true, "Overwrite previous output images, def false")
                << EAM(mPrefix,"Prefix",true, "Prefix for output images, def='8bits_'")
                << EAM(mVario,"Vario", true, "input files are Variocam infrateck camera frames, def=false")
                << EAM(mOptris,"Optris", true, "input files are Optris PI 640 camera frames, def=true")
                << EAM(mRGB,"RGB", true, "Export RGB images with color palette for thermic nice visualization, def false")
                << EAM(mDebug,"Debug", true, "Print messages in terminal to help debugging process, def false")
                );


    if (!MMVisualMode)
    {

        SplitDirAndFile(mDir,mPat,mFullDir);
        cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
        const std::vector<std::string> aSetIm = *(aICNM->Get(mPat));
        if (!EAMIsInit(&mOptris) && mVario==1) mOptris=0;

        Pt2di aRange;

        if (mVario==mOptris){
        std::cout << "Choose either Optris or Variocam image frame!\n";
        }

        if(mVario){
            // convert the range from deg to DN
            aRange.x=Deg2DN_Vario(mRangeT.x) ;
            aRange.y=Deg2DN_Vario(mRangeT.y) ;
        if (mDebug) std::cout << "Range of digital number on which stretching is performed (variocam images) : " << aRange << "\n";

        }
        if(mOptris){
            // for the optris images
            aRange.x=Deg2DN_Optris(mRangeT.x) ;
            aRange.y=Deg2DN_Optris(mRangeT.y) ;
            if (mDebug) std::cout << "Range of digital number on which stretching is performed (optris images) : " << aRange << "\n";
        }

        for (auto & im : aSetIm)
        {
            std::string NameOut(mDir+mPrefix+im);

            if (ELISE_fp::exist_file(NameOut) & !mOverwrite)
            {
                std::cout <<"Image " << NameOut <<" already exist, use F=1 to overwrite.\n";
            } else {

                // load input images
                Tiff_Im mTifIn=Tiff_Im::StdConvGen(mDir+im,1,true);
                // create empty RAM image for imput image
                Im2D_REAL4 imIn(mTifIn.sz().x,mTifIn.sz().y);
                // create empty RAM image for output image
                Im2D_U_INT1 imOut(mTifIn.sz().x,mTifIn.sz().y);
                // fill it with tiff image value
                ELISE_COPY(
                            mTifIn.all_pts(),
                            mTifIn.in(),
                            imIn.out()
                            );

                if (!mRGB){

                    int minRad(aRange.x), rangeRad(aRange.y-aRange.x);
                    // change radiometry and note min and max value
                    int aMin(255), aMax(0),aSum(0),aNb(0);
                    for (int v(0); v<imIn.sz().y;v++)
                    {
                        for (int u(0); u<imIn.sz().x;u++)
                        {
                            Pt2di pt(u,v);
                            double aVal = imIn.GetR(pt);
                            int val(0);

                            if(aVal!=0){
                                if (aVal>=minRad && aVal <minRad+rangeRad)
                                {
                                    val=255.0*(aVal-minRad)/rangeRad;
                                }
                                if (aVal >=minRad+rangeRad) val=255.0;
                            }
                            if (val>aMax) aMax=val;
                            if (val!=0){
                                if (val<aMin) aMin=val;
                                aSum+=val;
                                aNb++;
                            }
                            imOut.SetR(pt,val);
                            if (mDebug) std::cout << "aVal at position " << pt << " = " << aVal << ", I convert it to " << v <<"\n";
                        }
                    }
                    Tiff_Im aTifOut
                            (
                                NameOut.c_str(),
                                imOut.sz(),
                                GenIm::u_int1,
                                Tiff_Im::No_Compr,
                                Tiff_Im::BlackIsZero
                                );
                    std::cout << "Writing " << NameOut << ", Min " << aMin <<" Max "<< aMax <<" Mean "<< aSum/aNb <<  "\n";
                    ELISE_COPY(imOut.all_pts(),imOut.in(),aTifOut.out());


                //     RGB   export
                } else {

                    // RGB palette export
                    cISR_ColorImg  imRGB(imIn.sz());

                    for (int v(0); v<imIn.sz().y;v++)
                    {
                        for (int u(0); u<imIn.sz().x;u++)
                        {
                            Pt2di pt(u,v);
                            int aDN = imIn.GetR(pt);
                            double aDeg(0);
                            if (mOptris) aDeg=DN2Deg_Optris(aDN);
                            if (mVario) aDeg=DN2Deg_Vario(aDN);

                            cISR_Color col(Deg2R_pal1(aDeg,mRangeT), Deg2G_pal1(aDeg,mRangeT),Deg2B_pal1(aDeg,mRangeT));
                            imRGB.set(pt,col);
                        }
                    }
                    std::cout << "Writing " << NameOut << " as an RGB images\n";
                    imRGB.write(NameOut);
                }
            }
        }
    }
}


int ThermicTo8Bits_main(int argc,char ** argv)
{
   cThermicTo8Bits(argc,argv);
   return EXIT_SUCCESS;
}
