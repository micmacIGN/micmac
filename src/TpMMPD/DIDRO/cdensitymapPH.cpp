#include "cdensitymapPH.h"

// create a map of the entire area with very low resolution that depict the density of tie point (point homologue PH)
Im2D_REAL8 aDenoise(3,3,
                    "0 1 0 "
                    "1 2 1 "
                    " 0 1 0"
                    );


cDensityMapPH::cDensityMapPH(int argc,char ** argv)
{

    mOut="TiePoints_DensityMap.tif";
    mDebug=0;
    mExpTxt=0;
    mDir="./";
    mSmoothing=1;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()   << EAMC(mDir,"Working Directory", eSAM_IsDir)
                << EAMC(mOriPat,"Orientation (xml) list of file", eSAM_IsPatFile)
                ,
                LArgMain()  << EAM(mSH,"SH",true, "Set of Homol name")
                << EAM(mExpTxt,"ExpTxt",true, "Are tie points in txt format? default false, means standard binary format is used." )
                << EAM(mFileSH,"FileSH",true, "File of new set of homol format. If provided, argument as 'ExpTxt' and 'SH are not appropriated ",eSAM_IsExistFile )
                << EAM(mOut,"Out",true, "Name of resulting density map, default TiePoints_DensityMap( + SH).tif" )
                << EAM(mGSD,"GSD",true, "Ground Sample Distance of the resulting density map" )
                << EAM(mWidth,"Width",true, "Size [pix] of width resulting density map" )
                << EAM(mSmoothing,"Smooth",true, "Apply Gaussian filter to smooth the result, def true" )
                << EAM(mDebug,"Debug",true, "Print message in terminal to help debugging." )

                );

    if (!MMVisualMode)
    {
        mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);

        // load orientations
        mOriFL = mICNM->StdGetListOfFile(mOriPat);

        loadPH(); // load PH, convert old to new format is required

        std::string aKey= "NKS-Assoc-Ori2ImGen"  ;
        std::string aTmp1, aNameOri;

        for (auto &aOri : mOriFL){

            // retrieve name of image from name of orientation xml
            SplitDirAndFile(aTmp1,aNameOri,aOri);
            std::string NameIm = mICNM->Assoc1To1(aKey,aNameOri,true);
            mImName.push_back(NameIm);
            // retrieve IdIm
            cCelImTPM * ImTPM=mTPM->CelFromName(NameIm);
            if (ImTPM) {
            // map of Camera is idexed by the Id of Image (cSetTiePMul)
            mCams[ImTPM->Id()]=CamOrientGenFromFile(aOri,mICNM);
            } else {
            std::cout << "No tie points found for image " << NameIm << ".\n";
            }
        }

        determineMosaicFootprint();
        determineGSD();
        mDM=Im2D_REAL4(mSz.x,mSz.y,0.0);

        populateDensityMap();
        std::string aOut(mDir+mOut);
        if (mSmoothing) ELISE_COPY(mDM.all_pts(),som_masq(mDM.in_proj(),aDenoise)/6,mDM.out());
        Tiff_Im::CreateFromIm(mDM,aOut);
        if (mDebug) std::cout << "Result saved in " << aOut << ".\n";

    }
}

void cDensityMapPH::loadPH(){

    if (EAMIsInit(&mFileSH)){
        mTPM = new cSetTiePMul(0);
        //mTPM->SetFilter(mImName);
        mTPM->AddFile(mFileSH);
    }

    // old tie p format; load and convert??
    else {
        std::cout << "Warn, rigth now only new format of multi tie p is supporte, use FileSH argument.\n";

    }
}


void cDensityMapPH::determineMosaicFootprint(){

    if (mDebug) std::cout << "Determine Density Map footprint.\n";
    double xmin(3.4E+38),ymin(3.4E+38),xmax(0),ymax(0);
    bool first=true;
    for (auto & Cam : mCams)
    {
        double alt=Cam.second->GetAltiSol();
        Box2dr box=Cam.second->BoxTer(alt);
        if (first){
            xmin=box.P0().x;
            ymin=box.P0().y;
            xmax=box.P1().x;
            ymax=box.P1().y;
            first=false;
        } else {
            xmin=ElMin(xmin,box.P0().x);
            ymin=ElMin(ymin,box.P0().y);
            xmax=ElMax(xmax,box.P1().x);
            ymax=ElMax(ymax,box.P1().y);
        }
    }
    mBoxTerrain=Box2dr(Pt2dr(xmin,ymin),Pt2dr(xmax,ymax));
    if (mDebug) std::cout << "Ground box of density map is " << ToString(mBoxTerrain) << ", with means a swath of " << Pt2dr(mBoxTerrain.P1()-mBoxTerrain.P0()) <<" \n";
}

void cDensityMapPH::determineGSD(){

    if (!EAMIsInit(&mGSD) && !EAMIsInit(&mSz)) {
        double aGSDmean(0);
        for (auto& Cam : mCams){
            aGSDmean=aGSDmean+ Cam.second->GlobResol();
        }
        aGSDmean=aGSDmean/mCams.size();
        if (mDebug) std::cout << "mean GSD of images is " << aGSDmean << ".\n";
        // let's give 100 pixe per nadir image in ground geometry
        mGSD=aGSDmean*(((double)mCams[0]->Sz().x)/10.00);

        if (mDebug) std::cout << "Image size is" << ToString(mCams[0]->Sz()) << ".\n";
        if (mDebug) std::cout << "GSD computed for Density map is equal to " << mGSD << ".\n";
    } else if (EAMIsInit(&mWidth)) {
        mGSD= (mBoxTerrain.P1().x-mBoxTerrain.P0().x)/mWidth;
    }
    mSz= Pt2di(round_up((mBoxTerrain.P1().x-mBoxTerrain.P0().x)/mGSD),round_up((mBoxTerrain.P1().y-mBoxTerrain.P0().y)/mGSD));
    if (mDebug) std::cout << "Image size of density map is " << ToString(mSz) << ", which means GSD of " << mGSD <<"\n";
}

void cDensityMapPH::populateDensityMap(){

    // loop on every config of TPM of the set of TPM
    int progBar=ElMax(int(mTPM->VPMul().size()/10),1);
    int cnt(0);
    for (auto & aCnf : mTPM->VPMul())
    {
       // retrieve 3D position in model geometry
       std::vector<Pt3dr> aPts=aCnf->IntersectBundle(mCams);
       // add the points to the density map
       for (auto & Pt: aPts){

        Pt2dr PosXY(Pt.x,Pt.y);
        Pt2di PosUV=XY2UV(PosXY);
        //std::cout << "got a point " << PosXY << " , pos pixel : " << PosUV <<"\n";
        double aVal=mDM.GetR(PosUV);
        mDM.SetR(PosUV,aVal+1);
       }
     cnt++;
     if(cnt>progBar) {
         std::cout << "-";
         cnt=0;
     }
    }
    std::cout << "\n";
}

Pt2di cDensityMapPH::XY2UV(Pt2dr aVal){
    if (mBoxTerrain.contains(aVal))
    {
    Pt2di aRes((aVal.x-mBoxTerrain.P0().x)/mGSD,(aVal.y-mBoxTerrain.P0().y)/mGSD);
    return aRes;
    }   else {return Pt2di(0,0);}
}

int main_densityMapPH(int argc,char ** argv)
{
    cDensityMapPH(argc,argv);
    return EXIT_SUCCESS;
}
