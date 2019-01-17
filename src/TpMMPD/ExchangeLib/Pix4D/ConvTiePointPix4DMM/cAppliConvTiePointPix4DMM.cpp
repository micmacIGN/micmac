#include "ConvTiePointPix4DMM.h"


cPtMul::cPtMul(int aId_bingo):
    mID_bingo(aId_bingo)
{
}

ConvTiePointPix4DMM::ConvTiePointPix4DMM() :
    mSetTiep(new cSetTiePMul(0)),
    mCurImId(0),
    mBinOut(true),
    mImgFormat("")
{

}


bool ConvTiePointPix4DMM::IsPointRegisted(int aIdPoint)
{
    if(mMap_Id_PtMul.find(aIdPoint) != mMap_Id_PtMul.end())
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool ConvTiePointPix4DMM::IsImageRegisted(string aNameIm)
{
    if(mMap_IdImPmul_NameIm.find(aNameIm) != mMap_IdImPmul_NameIm.end())
    {
        return true;
    }
    else
    {
        return false;
    }
}


Pt2dr ConvTiePointPix4DMM::PtTranslation(Pt2dr aPt)
{
    return (Pt2dr(aPt.x + mCurSzIm_Bingo.x/2.0, -aPt.y + mCurSzIm_Bingo.y/2.0))*1000/mSzPhotosite;
}

int ConvTiePointPix4DMM::parseLine(string & aL, vector<string> & aVWord)
{
    if(aL.size() != 0)
    {
        //parse
        char *aBuffer = strdup((char*)aL.c_str());

        char * aWord;
        aWord = strtok (aBuffer," ");
        int aWordCounter = 0;
        while (aWord != NULL)
        {
           string aSWord(aWord);
           aVWord.push_back(aSWord);
           aWordCounter++;
           aWord = strtok (NULL, " ");
        }
        if (aWordCounter == 3 || aWordCounter == 4)
        {
            return A_POINT;
        }
        else if (aWordCounter == 2 || (aWordCounter == 1 && atoi(aVWord[0].c_str()) != -99) )
        {
            if (aWordCounter == 2)
            {
                string aSzIm_Bingo = aVWord[1];
                char * aWord1;
                char *aBuffer1 = strdup((char*)aSzIm_Bingo.c_str());
                aWord1 = strtok (aBuffer1,"_");
                while (aWord1 != NULL)
                  {
                    string aSWord(aWord1);
                    aVWord.push_back(aSWord);
                    aWord1 = strtok (NULL, "_");
                  }
            }
            return BEGIN_IMAGE;
        }
        else if (aWordCounter == 1 && atoi(aVWord[0].c_str()) == -99)
        {
            return END_IMAGE;
        }
        else
        {
            return UNKNOW_TYPE;
        }
        //Pt2dr aC(atof(aI),atof(aJ));
    }
    // Warning no return
    ELISE_ASSERT(false,"Should not be here");
    return 123456789;
}

bool ConvTiePointPix4DMM::ImportTiePointFile(string aFile, int & file_type)
{
    // detect automatiquement tiepoint format (Bingo ou Pix4D)
    mFileName = aFile;
    ifstream aF(aFile.c_str());
    if(!aF)
    {
        cout<<"File not exist ! Quit";
        return false;
    }
    else
    {
        std::string aL;
        getline(aF,aL,'\n'); // get first line to detect format
        vector<string> aVWord;
        int aType = parseLine(aL, aVWord);
        if (aType == BEGIN_IMAGE)
        {
            if (aVWord.size() == 1) // only image name
            {
                cout<<"Pix4D file type"<<endl;;
                file_type = _TP_PIX4D;
            }
            if (aVWord.size() == 5) // imageName SizeCapteur SizeCapteurX SizeCapteurY
            {
                cout<<"Bingo file type"<<endl;;
                file_type = _TP_BINGO;
            }
            if (aVWord.size() == 6) // imageName IdPoint X(mm) Y(mm) 0 M
            {
                 cout<<"Orima file type"<<endl;;
                 file_type = _TP_ORIMA;
            }
            return true;
        }
        else
        {
            cout<<"Unknow file type"<<endl;;
            file_type = UNKNOW_TYPE;
            return false;
        }
    }
}

bool ConvTiePointPix4DMM::ReadPix4DFile(string aFile)
{
    mFileName = aFile;
    bool hasImgFormat = false;
    ifstream aF(aFile.c_str());
    if(!aF)
    {
        return false;
    }
    else
    {
        std::string aL;
        int aCntPt = 0;
        while(!aF.eof())
        {
            getline(aF,aL,'\n');
            vector<string> aVWord;
            int aType = parseLine(aL, aVWord);
            if (aType == BEGIN_IMAGE)
            {
                mCurIm = aVWord[0];
                if(hasImgFormat == false)
                {
                     std::size_t found = mCurIm.find_first_of(".");
                     if (found!=std::string::npos)
                     {
                         hasImgFormat = true;
                     }
                     else
                     {
                         cout<<"WARN : image name in Pix4D file doesn't have extension."<<endl;
                         cout<<"Please type image extension (for example : .JPG, .tif ...): "<<endl;
                         string aGetExt;
                         cin>>aGetExt;
                         if(aGetExt.find_first_of(".") == std::string::npos)
                         {
                             aGetExt = "." + aGetExt;
                             cout<<"WARN : file extension input without \".\" -- auto-complete : "<<aGetExt<<endl;
                         }
                         mImgFormat = aGetExt;
                         hasImgFormat = true;
                     }
                }

                if (IsImageRegisted(mCurIm) == false)
                {
                    mMap_IdImPmul_NameIm.insert(std::make_pair(mCurIm, mCurImId));
                    mCurImId++;
                }
            }
            if (aType == A_POINT)
            {
                int aIdPoint = atoi(aVWord[0].c_str());
                Pt2dr aPtOrg(atof(aVWord[1].c_str()),atof(aVWord[2].c_str()));
                Pt2dr aPtMM = aPtOrg;
                if (IsPointRegisted(aIdPoint) == false)
                {
                    cPtMul * aPMul = new cPtMul(aIdPoint);
                    mMap_Id_PtMul.insert(std::make_pair(aIdPoint, aPMul));
                    std::map<string, int>::iterator it2;
                    it2 = mMap_IdImPmul_NameIm.find(mCurIm);
                    int idIm = it2->second;
                    aPMul->VIm().push_back(idIm);
                    aPMul->VPt().push_back(aPtMM);
                    aCntPt++;
                }
                else
                {
                    std::map<int, cPtMul*>::iterator it;
                    it = mMap_Id_PtMul.find(aIdPoint);
                    cPtMul * aPMul = it->second;
                    std::map<string, int>::iterator it2;
                    it2 = mMap_IdImPmul_NameIm.find(mCurIm);
                    int idIm = it2->second;
                    aPMul->VIm().push_back(idIm);
                    aPMul->VPt().push_back(aPtMM);
                }
            }
        }
        cout<<"Total Pix4d Tie Points : "<<aCntPt<<endl;
    }
    return true;
}

bool ConvTiePointPix4DMM::ReadBingoFile(string aFile)
{
    mFileName = aFile;
    bool hasImgFormat = false;
    ifstream aF(aFile.c_str());
    if(!aF)
    {
        return false;
    }
    else
    {
        std::string aL;
        int aCntPt = 0;
        while(!aF.eof())
        {
            getline(aF,aL,'\n');
            vector<string> aVWord;
            int aType = parseLine(aL, aVWord);
            if (aType == BEGIN_IMAGE)
            {
                mCurIm = aVWord[0];

                if(hasImgFormat == false)
                {
                     std::size_t found = mCurIm.find_first_of(".");
                     if (found!=std::string::npos)
                     {
                         hasImgFormat = true;
                     }
                     else
                     {
                         cout<<"WARN : image name in Pix4D Bingo file doesn't have extension."<<endl;
                         cout<<"Please type image extension (for example : .JPG, .tif ...): "<<endl;
                         string aGetExt;
                         cin>>aGetExt;
                         if(aGetExt.find_first_of(".") == std::string::npos)
                         {
                             aGetExt = "." + aGetExt;
                             cout<<"WARN : file extension input without \".\" -- auto-complete : "<<aGetExt<<endl;
                         }
                         mImgFormat = aGetExt;
                         hasImgFormat = true;
                     }
                }

                mCurSzIm_Bingo = Pt2dr(atof(aVWord[3].c_str()), atof(aVWord[4].c_str()));
                if (IsImageRegisted(mCurIm) == false)
                {
                    mMap_IdImPmul_NameIm.insert(std::make_pair(mCurIm, mCurImId));
                    mCurImId++;
                }
            }
            if (aType == A_POINT)
            {
                int aIdPoint = atoi(aVWord[0].c_str());
                Pt2dr aPtBingo(atof(aVWord[1].c_str()),atof(aVWord[2].c_str()));
                Pt2dr aPtMM = PtTranslation(aPtBingo);
                if (IsPointRegisted(aIdPoint) == false)
                {
                    cPtMul * aPMul = new cPtMul(aIdPoint);
                    mMap_Id_PtMul.insert(std::make_pair(aIdPoint, aPMul));
                    std::map<string, int>::iterator it2;
                    it2 = mMap_IdImPmul_NameIm.find(mCurIm);
                    int idIm = it2->second;
                    aPMul->VIm().push_back(idIm);
                    aPMul->VPt().push_back(aPtMM);
                    aCntPt++;
                }
                else
                {
                    std::map<int, cPtMul*>::iterator it;
                    it = mMap_Id_PtMul.find(aIdPoint);
                    cPtMul * aPMul = it->second;
                    std::map<string, int>::iterator it2;
                    it2 = mMap_IdImPmul_NameIm.find(mCurIm);
                    int idIm = it2->second;
                    aPMul->VIm().push_back(idIm);
                    aPMul->VPt().push_back(aPtMM);
                }
            }
        }
        cout<<"Total Pix4d Tie Points : "<<aCntPt<<endl;
    }
    return true;
}

void ConvTiePointPix4DMM::exportToMMNewFH()
{
    std::map<int, cPtMul*>::iterator it = mMap_Id_PtMul.begin();
    std::map<string, int>::iterator it2 = mMap_IdImPmul_NameIm.begin();
    vector<string> aNameIm (mMap_IdImPmul_NameIm.size());

    while(it2 != mMap_IdImPmul_NameIm.end())
    {
        aNameIm[it2->second] = string(it2->first + mImgFormat);
        it2++;
    }
    mSetTiep = new cSetTiePMul (0, &aNameIm);

    while(it != mMap_Id_PtMul.end())
    {
        cPtMul * aPt = it->second;
        vector<float> vAttr;
        mSetTiep->AddPts(aPt->VIm(), aPt->VPt(), vAttr);
        it++;
    }

    std::string aSaveName = "PMul" + mSuffixOut + (mBinOut ? ".dat" : ".txt");
    mSetTiep->Save(aSaveName);
    cout<<"Convert from : " << mFileName <<" to MicMac tie point : "<<aSaveName<<endl<<endl;
    mPMulFileName = aSaveName;
}

void ConvTiePointPix4DMM::exportToMMClassicFH(string aDir, string aOut, bool aBin, bool aIs2Way)
{

    cAppliConvertToOldFormatHom aConv(aDir , mPMulFileName , mSuffixOut, mBinOut, aIs2Way);
}

/*
    Vector bool contain flag to know if an id point is registed
    Read Bingo file :
        Each Id Point  :
            Check if point is registed (by bool vector or by map search)
            If not:
                Create aPMul object, then registed it
                Add aPMul object to map with its Id in a global map

*/
