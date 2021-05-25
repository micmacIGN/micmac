#include "ctapioca_idr.h"
std::string  TypeTapioca[4] = {"MulScale","All","Line","File"};

cTapioca_IDR::cTapioca_IDR(int argc, char** argv):
    mRatio(0.6),
    mIsSFS(0),
    mExpTxt(0),
    mPurge(1),
    mSH_post("-IDR"),
    mTmpDir("Tmp-TapiocaIDR"),
    mMergeHomol(1),
    mDebug(0),
    mHomolIDR("Homol" + mSH_post),
    mHomolFormat("dat")
{

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(mMode, "Tapioca mode")
                << EAMC(mPatOrFile, "Image Pattern (or file of images couples for File mode)")
                << EAMC(mImLengthOut, "Image length for tie point computation, in pixel"),
                LArgMain()  << EAM(mIsSFS,"SFS", true, "Apply SFS filter prior to TP computation" )
                << EAM(mLowRes,"LowRes",true,"Low resolution of images for mode MulScale of tapioca")
                << EAM(mNbNb,"NbNb",true,"NumBer of NeighBours for mode Line of tapioca")
                << EAM(mExpTxt,"ExpTxt",true,"Export TP to file format?")
                << EAM(mPurge,"Purge",true,"Purge temporary files? def true")
                << EAM(mTmpDir,"Dir",true,"Directory of temporary files, def Tmp-TapiocaIDR/")
                << EAM(mMergeHomol,"MergeSH",true,"Merge the resulting TP with current Homol/ set? Default true")
                << EAM(mSH_post,"PostFix",true,"Postfix for resulting Homol directory, default '-IDR' result in Homol-IDR/")
                << EAM(mDebug,"Debug",true,"Display terminal message? Default false")
                << EAM(mDetect,"Detect",true,"Detector tool")
                << EAM(mRatio,"Ratio",true,"Ann closeness ratio")
                );
    if (ELISE_fp::IsDirectory(mTmpDir)){
    std::cout << "Purge of temporary directory " << mTmpDir << "\n";
    ELISE_fp::PurgeDirGen(mTmpDir,1);
    }

    bool goOn(1);

    if(!ELISE_fp::IsDirectory(mTmpDir)) ELISE_fp::MkDir(mTmpDir);

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
            mICNM=cInterfChantierNameManipulateur::BasicAlloc("./");
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

    if (EAMIsInit(&mSH_post))  mHomolIDR = "Homol" + mSH_post;
    if (EAMIsInit(&mExpTxt) & mExpTxt) mHomolFormat="txt";


    if (goOn) {

        std::cout << "Resize Images\n";
        resizeImg();
    }

    goOn = runTapioca();

    if (goOn){
        std::cout << "Scale Tie Point pack\n";
        scaleTP();

        if (mMergeHomol)
        {
            std::cout << "Merge Tie Point pack\n";
            mergeHomol();
        }

        if (mPurge) {
            std::cout << "Purge of temporary directory " << mTmpDir << "\n";
            ELISE_fp::PurgeDirGen(mTmpDir,1);
        }
    }
    std::cout << "End of Tapioca with Images of Different Resolution.\n";
}

void cTapioca_IDR::scaleTP()
{

    std::list<std::string> aLCom;

    // create Homol-IDR

    if (ELISE_fp::IsDirectory(mHomolIDR)) ELISE_fp::PurgeDirGen(mHomolIDR,1);
    ELISE_fp::MkDir(mHomolIDR);

    // accosication key, two of them must be in localchantierdescripteur ('WithDir' one)
    std::string aKeyAsocHomWithDir = "NKS-Assoc-CplIm2HomWithDir@@" + mHomolFormat +"@" +mTmpDir;
    std::string aKeyAsocHom = "NKS-Assoc-CplIm2Hom@"+ mSH_post +"@" + mHomolFormat +"@" +mTmpDir;

    // loop on mName1 of mLIms
    for (auto & im1 : mLFile)
    {
    std::string aKeySetHom = "NKS-Set-HomolOfOneImageWithDir@@" + mHomolFormat +"@" + im1 +"@" +mTmpDir  ;

    const std::vector<std::string>* aVH = mICNM->Get(aKeySetHom);
    for (int aKH = 0 ; aKH <int(aVH->size()) ; aKH++)
    {

         std::string im2 =  mICNM->Assoc2To1(aKeyAsocHomWithDir,(*aVH)[aKH],false).second;
         std::string aHomolNameOut =  mICNM->Assoc1To2(aKeyAsocHom,im1,im2,true);

         std::string aCom =    MMBinFile(MM3DStr) + " TestLib ResizeHomol "
                 + (*aVH)[aKH] + " "
                 + ToString(mImRatio[im1]) + " "
                 + ToString(mImRatio[im2]) + " "
                 + aHomolNameOut
                 + "  Print=" + ToString(mDebug)
                 ;
         aLCom.push_back(aCom);
         if (mDebug) std::cout << aCom << "\n";
    }
    }
    cEl_GPAO::DoComInParal(aLCom);
}

void cTapioca_IDR::mergeHomol()
{
        std::string aCom =    MMBinFile(MM3DStr) + " MergeHomol "
                    + "'(Homol|Homol" + mSH_post + ")'" + "   Homol PurgeOut=0"
                ;
        if (mDebug) std::cout << aCom << "\n";
        VoidSystem(aCom.c_str());
}


void cTapioca_IDR::resizeImg()
{
    std::list<std::string> aLCom;
    for (auto & im : mLFile)
    {

        // fulfill map , key= images name, mapped value: ratio of changing size
        // is used later to scale tie point pack
        Tiff_Im tiff = Tiff_Im::BasicConvStd(im.c_str());
        mImRatio[im]= tiff.sz().x/(double)mImLengthOut;

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
    if (EAMIsInit(&mExpTxt)) aCom+= " ExpTxt=" + ToString(mExpTxt);
    if (EAMIsInit(&mIsSFS) & mIsSFS) aCom+= " @SFS";
    if (EAMIsInit(&mDetect)) aCom+= " Detect="+mDetect;
    if (EAMIsInit(&mRatio)) aCom+= " Ratio="+ToString(mRatio);

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

    if (sizeOutOK & !mF){
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



cResizeHomol::cResizeHomol(int argc, char** argv)
{
    bool aDebug(0);
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(mHomolNameIn, "Pack of homol In",eSAM_IsExistFile)
                << EAMC(mR1,"Scaling ratio for image 1")
                << EAMC(mR2,"Scaling ratio for image 2")
                << EAMC(mHomolNameOut, "Pack of homol Out"),
                LArgMain()
                << EAM(aDebug,"Print",true,"Print msg to console for inspection, def false")
                );
   // read homol
   ElPackHomologue aPack = ElPackHomologue::FromFile(mHomolNameIn);

   if (aDebug)
   {
   ElCplePtsHomologues aTPIn= aPack.Cple_Back();
   ElCplePtsHomologues aTPOut(aTPIn);
   aTPOut.P1()= aTPOut.P1()*mR1;
   aTPOut.P2()= aTPOut.P2()*mR2;
   std::cout << "Resize Homol with ratio " << mR1 << "," << mR2 << "\n";
   std::cout << "tie points " << aTPIn.P1() << "," << aTPIn.P2() << " become \n " << aTPOut.P1() << "," <<aTPOut.P2() << "\n";
   }
   // resize homol with ratio
   aPack.Resize(mR1,mR2);
   // write homol
   aPack.StdAddInFile(mHomolNameOut);

}


int resizeHomol_main(int argc,char ** argv)
{
    cResizeHomol(argc,argv);
    return EXIT_SUCCESS;
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





