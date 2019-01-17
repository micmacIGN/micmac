#include "tiepgraphbycamdist.h"

cAppliTiepGraphByCamDist::cAppliTiepGraphByCamDist() :
    mInv (false),
    mOriMode (false)
{}

void cAppliTiepGraphByCamDist::ImportNavInfo(string & aName)
{
    if (mOriMode == true)
    {
        cout<<"Ori Folder mode"<<endl;
        ImportMicMacOriFolder(aName);
    }
    else
    {
        cout<<"Nav file mode"<<endl;
        ImportOri(aName);
    }
}


int cAppliTiepGraphByCamDist::parseLine(string & aL, vector<string> & aVWord)
{
    if(aL.size() != 0)
    {
        //parse
        char *aBuffer = strdup((char*)aL.c_str());
        char * aWord;
        aWord = strtok (aBuffer," \t;,");
        int aWordCounter = 0;
        while (aWord != NULL)
        {
           string aSWord(aWord);
           aVWord.push_back(aSWord);
           aWordCounter++;
           aWord = strtok (NULL," \t;,");
        }
        return aWordCounter;
    }
    else
        return LINE_EMPTY;
}


bool cAppliTiepGraphByCamDist::ImportOri(string & aFName)
{
    ifstream aF(aFName.c_str());
    if(!aF)
    {
        cout<<"File not exist ! Quit";
        return false;
    }
    else
    {
        int aLineCnt = 0;
        while(!aF.eof())
        {
            std::string aL;
            getline(aF,aL,'\n'); // get first line to detect format
            vector<string> aVWord;
            int aType = parseLine(aL, aVWord);
            aLineCnt++;
            if (aType != LINE_EMPTY)
            {
                if (aVWord.size() < 4)
                {
                    cout << "Not enough 4 Column ! Skip this line, must contains at least 4 comumn : N(name_image) X Y Z"<<endl;
                    cout<<aL<<endl;
                }
                else
                {
                    // get data :
                    string aNameIM = aVWord[0];
                    double aX = atof(aVWord[1].c_str());
                    double aY = atof(aVWord[2].c_str());
                    double aZ = atof(aVWord[3].c_str());
                    cOneImg * aIm = new cOneImg(aNameIM, Pt3dr(aX, aY, aZ));
                    if (aVWord.size() > 6)
                    {
                        double aW = atof(aVWord[4].c_str());
                        double aP = atof(aVWord[5].c_str());
                        double aK = atof(aVWord[6].c_str());
                        aIm->mOri = Pt3dr(aW, aP, aK);
                    }
                    mMap_NameIm_NavInfo.insert(std::make_pair(aNameIM, aIm));
                    mVNameIm.push_back(aNameIM);
                    //cout.precision(17);
                    //cout<<" Im : "<<aIm->mName<<" "<<aIm->mPt<<" "<<aIm->mOri<<endl;
                }
            }
        }
    }
    return true; // MPD :Warnig no return value
}

void cAppliTiepGraphByCamDist::ImportMicMacOriFolder(string & aFolderName)
{
    // check if Ori Dir has "/" at the end
    if ( aFolderName.find(ELISE_STR_DIR) != aFolderName.length()-1)
    {
        cout<<"Correction Ori folder name"<<endl;
        aFolderName = aFolderName + ELISE_STR_DIR;
    }

    string aOriXMLPat = "Orientation-.*.xml";

    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aFolderName);

    vector<string> aSetOri = *(aICNM->Get(aOriXMLPat));

    ELISE_ASSERT(aSetOri.size()>0,"Can't get any orientation file (Orientation-*.xml)");

    cout<<"Nb Ori file get : "<<aSetOri.size()<<endl;

    // Get image name from Orientation file name
    for (uint aKOri = 0; aKOri < aSetOri.size(); aKOri++)
    {
        string aNameIm = aICNM->Assoc1To1("NKS-Assoc-Ori2ImGen", aSetOri[aKOri], 1);   // cle compute de orientation name to image name
        CamStenope * aCam = CamOrientGenFromFile(aSetOri[aKOri] , aICNM);
        cOneImg * aIm = new cOneImg(aNameIm, aCam->VraiOpticalCenter());
        mMap_NameIm_NavInfo.insert(std::make_pair(aNameIm, aIm));
        mVNameIm.push_back(aNameIm);
        cout<<"Im : "<<aNameIm<<endl;
    }
}

void cAppliTiepGraphByCamDist::ComputeImagePair(double aD)
{
    // parcour tout images dans la list, calcul distance avec tout les autres.
    std::map<string, cOneImg*>::iterator it_MapFind;
    it_MapFind = mMap_NameIm_NavInfo.end();
    for (uint aKIm = 0; aKIm < mVNameIm.size(); aKIm++)
    {
        string aNameIm = mVNameIm[aKIm];
        it_MapFind = mMap_NameIm_NavInfo.find(aNameIm);
        if (it_MapFind != mMap_NameIm_NavInfo.end())
        {
            cOneImg * aIm1 = it_MapFind->second;
            for (uint aKIm2 = aKIm+1; aKIm2 < mVNameIm.size(); aKIm2++)
            {
                string aNameIm2 = mVNameIm[aKIm2];
                it_MapFind = mMap_NameIm_NavInfo.find(aNameIm2);
                if (it_MapFind != mMap_NameIm_NavInfo.end())
                {
                    cOneImg * aIm2 = it_MapFind->second;
                    Pt3dr aVecDist = (aIm1->mPt - aIm2->mPt).AbsP();
                    double dist = sqrt(aVecDist.x*aVecDist.x + aVecDist.y*aVecDist.y + aVecDist.z*aVecDist.z);
                    if (dist <= aD)
                    {
                        //cout<<aNameIm<<" "<<aNameIm2<<" "<<dist<<endl;
                        // Create image pair
                        mRelXML.Cple().push_back(cCpleString(aIm1->mName, aIm2->mName));
                        if (mInv)
                        {
                            mRelXML.Cple().push_back(cCpleString(aIm2->mName, aIm1->mName));
                        }
                    }
                }
            }
        }
        else
            continue;
    }
}

void cAppliTiepGraphByCamDist::ExportGraphFile(string & aFName)
{
    MakeFileXML(mRelXML,aFName);
}

std::string BannerTiepGraphByCamDist()
{
    std::string banniere = "\n";
    banniere += "************************************************************************* \n";
    banniere += "**                 Generate an image pair for tie point search,        ** \n";
    banniere += "**                 constraint by distance between camera center        ** \n";
    banniere += "**                       (given by GPS or anything else)               ** \n";
    banniere += "**  1) User give Ori folder or Navigation text file                    ** \n";
    banniere += "**     (format N_X_Y_Z_W_P_K as file for OriConvert tool)              ** \n";
    banniere += "**  2) User gives Average distance between consecutive images          ** \n";
    banniere += "**  3) Tools will create a sphere with center is this image center and ** \n";
    banniere += "**     radius of Average distance, than generate match couples of this ** \n";
    banniere += "**     image with all other images included in the sphere              ** \n";
    banniere += "************************************************************************* \n";
    return banniere;
}

int TiepGraphByCamDist_main(int argc,char ** argv)
{

    cout<<BannerTiepGraphByCamDist();
    string aNavFile;
    string aOriFolder;
    string aXMLGrapheOut = "ImgPairByDist.xml";
    double aDist = double(-1.0);
    bool aInv = false;
    string aPat;

    ElInitArgMain
    (
          argc, argv,
          LArgMain(),
          LArgMain()
                << EAM(aNavFile,"Nav",false,"GPS Navigation file (text file, format : at least 4 Columns : NameIm X Y Z")
                << EAM(aOriFolder,"Ori",false,"Orientation folder (if don't have nav file)")
                << EAM(aXMLGrapheOut,"Out",false,"Image Pair output file (to be computed with Tapioca File - def = ImgPairByDist.xml)")
                << EAM(aDist,"D",false,"Maximum distance to be an image pair")
                << EAM(aInv,"Inv",false,"Do you want to add an inverse for each couple created ? (for ex, Im1-Im2 -> add Im2-Im1")
                << EAM(aPat,"Pat",false,"Pattern of image. Generate with all images in navigation file if user not enter")
    );

    cAppliTiepGraphByCamDist * aAppli = new cAppliTiepGraphByCamDist();
    cout<<EAMIsInit(&aNavFile) << " "<<EAMIsInit(&aOriFolder)<<endl;
    if((EAMIsInit(&aNavFile) && EAMIsInit(&aOriFolder)))
    {
        cout<<"Please give Nav file by Nav option OR Ori folder by Ori option (1 option only). type -help for all command option"<<endl;
        return EXIT_FAILURE;
    }
    if(!(EAMIsInit(&aNavFile) || EAMIsInit(&aOriFolder)))
    {
        cout<<"Please give Nav file by Nav option or Ori folder by Ori option. type -help for all command option"<<endl;
        return EXIT_FAILURE;
    }
    if(!EAMIsInit(&aDist))
    {
        cout<<"Please give max distance to search for image pair by D option. type -help for all command option"<<endl;
        return EXIT_FAILURE;
    }
    if((EAMIsInit(&aNavFile) || EAMIsInit(&aOriFolder)))
    {
        if ((EAMIsInit(&aNavFile)))
        {
             cout<<"Navigation file given : "<<aNavFile<<endl;
             if (!ELISE_fp::exist_file(aNavFile.c_str()))
             {
                 cout<<"Navigation file not founded ! "<<endl;
                 return EXIT_FAILURE;
             }
             aAppli->OriMode() = false;
             aAppli->ImportNavInfo(aNavFile);
        }
        if ((EAMIsInit(&aOriFolder)))
        {
             cout<<"Ori folder given : "<<aOriFolder<<endl;
             if (!ELISE_fp::IsDirectory(aOriFolder))
             {
                 cout<<"Ori folder not founded ! "<<endl;
                 return EXIT_FAILURE;
             }
             aAppli->OriMode() = true;
             aAppli->ImportNavInfo(aOriFolder);
        }
    }
    aAppli->Inv() = aInv;
    aAppli->ComputeImagePair(aDist);
    aAppli->ExportGraphFile(aXMLGrapheOut);

    cout<<endl<<endl<<"********  Finish  **********"<<endl;
    return EXIT_SUCCESS;
}
