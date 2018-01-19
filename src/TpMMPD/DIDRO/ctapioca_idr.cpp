#include "ctapioca_idr.h"
std::string  TypeTapioca[4] = {"MulScale","All","Line","File"};

cTapioca_IDR::cTapioca_IDR(int argc, char** argv):mIsSFS(0),mExpTxt(0),mPurge(1),mSH_post("_IDR"),mTmpDir("Tmp-TapiocaIDR"),mMergeHomol(1),mDebug(1)
{

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(mMode, "Tapioca mode, if File, must specify the file with File optionnal argument")
                << EAMC(mPatOrFile, "Image Pattern or file of images couples")
                << EAMC(mImLengthOut, "Image length for tie point computation, in pixel"),
                LArgMain()  << EAM(mIsSFS,"SFS", true, "Apply SFS filter prior to TP computation" )
                << EAM(mLowRes,"LowRes",true,"Low resolution of images for mode MulScale of tapioca")
                << EAM(mNbNb,"NbNb",true,"NumBer of NeighBours for mode Line of tapioca")
                << EAM(mExpTxt,"ExpTxt",true,"Export TP to file format?")
                << EAM(mPurge,"Purge",true,"Purge temporary files? def true")
                << EAM(mTmpDir,"Dir",true,"Directory of temporary files, def Tmp-TapiocaIDR/")
                << EAM(mMergeHomol,"MergeSH",true,"Merge the resulting TP with current Homol/ set? Default true")
                << EAM(mDebug,"Debug",true,"Display terminal message? Default false")
                );

    bool goOn(1);

    if ((mMode==TypeTapioca[0]) | (mMode==TypeTapioca[1]) | (mMode==TypeTapioca[2])) {
        SplitDirAndFile(mDir,mImPat,mPatOrFile);
        // define the "working directory" of this session
        mICNM=cInterfChantierNameManipulateur::BasicAlloc(mDir);
        // create the list of images starting from the regular expression (Pattern)
        mLFile = mICNM->StdGetListOfFile(mImPat);

        std::cout << "Dir: " << mDir << ", ImPat :" << mImPat << ", " << mLFile.size() << " element found.\n";

    } else if (mMode==TypeTapioca[3]){
        // mode file

        if (ELISE_fp::exist_file(mPatOrFile)  )
        {
            // récupère une liste d'image
            cSauvegardeNamedRel aSNR = StdGetFromPCP(mPatOrFile,SauvegardeNamedRel);
            for
                    (
                     std::vector<cCpleString>::const_iterator itC=aSNR.Cple().begin();
                     itC!=aSNR.Cple().end();
                     itC++
                     )
            {

                bool find = std::find(mLFile.begin(), mLFile.end(), itC->N1()) != mLFile.end() ;
                if (!find) mLFile.push_back(itC->N1());
                find = std::find(mLFile.begin(), mLFile.end(), itC->N2()) != mLFile.end() ;
                if (!find) mLFile.push_back(itC->N2());
            }
            std::cout << "\nImages File :" << mPatOrFile << ", " << mLFile.size() << " images found.\n\n";
            // copy the file to the directory of tapioca IDR
            MakeFileXML(aSNR,mTmpDir+"/"+mPatOrFile);

        } else {
            std::cout << "Error : cannot find the following file of images couples : " << mPatOrFile << " .\n" ;
            goOn=0;
        }

    }  else {
        std::cout << "Mode " << mMode << " not supported. Try MulScale, All, Line or File.\n" ;
        goOn=0;
    }


    if (goOn) {

    if(!ELISE_fp::IsDirectory(mTmpDir)) ELISE_fp::MkDir(mTmpDir);
    resizeImg();
    }

    goOn = runTapioca();

    if (goOn){

        if (mMergeHomol)
        {


        }
    }
}

void cTapioca_IDR::resizeImg()
{
    std::list<std::string> aLCom;
    for (auto & im : mLFile)
    {
        std::string aCom =    MMBinFile(MM3DStr) + " TestLib ResizeImg "
                + mDir + im
                + "  "
                + mTmpDir + "/" + im
                + "  "
                + ToString(mImLengthOut)
                ;
        aLCom.push_back(aCom);
        if (mDebug) std::cout << aCom << "\n";
    }
    cEl_GPAO::DoComInParal(aLCom);
}

int cTapioca_IDR::runTapioca()
{


    std::string aCom(" ");
    if (mMode==TypeTapioca[0]){

        if (EAMIsInit(&mLowRes))
        {
            aCom =    MMBinFile(MM3DStr) + "Tapioca MulScale "
                    + "'" + mTmpDir + "/" + mImPat + "' "
                    + ToString(mImLengthOut) + " "
                    + ToString(mLowRes) + " "
                    ;
        } else {
            std::cout << "Error : Mode MulScale require to provide a value for optionnal argument 'LowRes'.\n" ;
            return 0;
        }

    } else if (mMode==TypeTapioca[1]){
        aCom =    MMBinFile(MM3DStr) + "Tapioca All "
                + "'" + mTmpDir + "/" + mImPat + "' "
                + ToString(mImLengthOut) + " "
                ;
    } else if (mMode==TypeTapioca[2]){
        if (EAMIsInit(&mNbNb))
        {
            aCom =    MMBinFile(MM3DStr) + "Tapioca Line "
                    + "'" + mTmpDir + "/" + mImPat + "' "
                    + ToString(mImLengthOut) + " "
                    + ToString(mNbNb) + " "
                    ;
        } else {
            std::cout << "Error : Mode Line require to provide a value for optionnal argument 'NbNb'.\n" ;
            return 0;
        }
    } else if (mMode==TypeTapioca[3]){

        aCom =    MMBinFile(MM3DStr) + "Tapioca File "
                + mTmpDir + "/" + mPatOrFile + " "
                + ToString(mImLengthOut) + " "
                ;
    }
    if (EAMIsInit(&mExpTxt)) aCom=+ " ExpTxt=" + mExpTxt;
    if (EAMIsInit(&mIsSFS) & mIsSFS) aCom=+ " @SFS";

    if (mDebug) std::cout << "Tapioca command: " << aCom << "\n";

    VoidSystem(aCom.c_str());

    return 1;
}






cResizeImg::cResizeImg(int argc, char** argv):mF(0)
{
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(mImNameIn, "Image Name, tif format",eSAM_IsExistFile)
                << EAMC(mImNameOut, "Output Name")
                << EAMC(mImLengthOut, "Image length out, in pixel"),
                LArgMain()
                << EAM(mF,"F",true,"Force Overwriting Image Out if it already exist and have the size you want? default false")
                );

    Tiff_Im tiff = Tiff_Im::BasicConvStd(mImNameIn.c_str());
    mImSzIn=tiff.sz();
    // even if there are any rounding approximation, the image ratio is kept hopefully
    mImSzOut=Pt2di(mImLengthOut,mImSzIn.y*mImLengthOut/mImSzIn.x);

    bool sizeOutOK(0);

    // test if output image have the resolution we want to resize to
    if (ELISE_fp::exist_file(mImNameOut))
    {
        Tiff_Im tiffOut = Tiff_Im::BasicConvStd(mImNameOut.c_str());
        if (mImSzOut==tiffOut.sz()) sizeOutOK=1;
    }

    if (sizeOutOK | !mF){
        std::cout << "Do not ResizeIm because output images already exist and is properly resized, use F=1 to force overwriting.\n";
    } else {
    std::cout << "ResizeIm use convert (http://doc.ubuntu-fr.org/imagemagick)\n";
    std::string  aCom =     std::string("convert ")
            +   mImNameIn +  std::string(" ")
            +   std::string(" -resize ")
            +   ToString(mImSzOut.x) + "x" + ToString(mImSzOut.y)
            +   std::string(" ") + mImNameOut;
    VoidSystem(aCom.c_str());
    }
}

int resizeImg_main(int argc,char ** argv)
{
    cResizeImg(argc,argv);
    return EXIT_SUCCESS;
}

int Tapioca_IDR_main(int argc,char ** argv)
{
    cTapioca_IDR(argc,argv);
    return EXIT_SUCCESS;
}



