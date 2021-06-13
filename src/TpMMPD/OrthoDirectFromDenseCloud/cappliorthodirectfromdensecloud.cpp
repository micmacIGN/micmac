#include "orthodirectfromdensecloud.h"

class cAppliOrthoDirectFromDenseCloud;
// rectification directly point cloud to plan image

typedef double tPxl;
typedef Im2D<tPxl,tPxl>  tImOrtho;
typedef TIm2D<tPxl,tPxl> tTImOrtho;
typedef Im2D_U_INT1 tImOrthoUINT;
#define Z_DEFAULT -10000.0
typedef cInterpolateurIm2D<tPxl>  tInterpolFillHole;


typedef struct sPlyColoredOrientedVertex64
{
    float x, y, z;
    unsigned char red, green, blue;
    float nx, ny, nz;
} sPlyColoredOrientedVertex64;

static PlyProperty colored_oriented_vert_props[] = {
    {"x",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredOrientedVertex64,x ), 0, 0, 0, 0},
    {"y",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredOrientedVertex64,y ), 0, 0, 0, 0},
    {"z",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredOrientedVertex64,z ), 0, 0, 0, 0},
    {"red",   PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredOrientedVertex64,red), 0, 0, 0, 0},
    {"green", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredOrientedVertex64,green), 0, 0, 0, 0},
    {"blue",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredOrientedVertex64,blue), 0, 0, 0, 0},
    {"nx", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredOrientedVertex64,nx), 0, 0, 0, 0},
    {"ny", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredOrientedVertex64,ny), 0, 0, 0, 0},
    {"nz", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredOrientedVertex64,nz), 0, 0, 0, 0},
};



typedef struct sPlyPix4D
{
    float x, y, z;
    unsigned char red, green, blue;
} sPlyPix4D;

static PlyProperty props_Pix4D[] = {
    {"x",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyPix4D,x ), 0, 0, 0, 0},
    {"y",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyPix4D,y ), 0, 0, 0, 0},
    {"z",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyPix4D,z ), 0, 0, 0, 0},
    {"diffuse_red",   PLY_UCHAR, PLY_UCHAR, offsetof(sPlyPix4D,red), 0, 0, 0, 0},
    {"diffuse_green", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyPix4D,green), 0, 0, 0, 0},
    {"diffuse_blue",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyPix4D,blue), 0, 0, 0, 0},
};


class cVertex3D
{
    public :
        cVertex3D(double & x, double & y, double & z, double & R, double & G, double & B);
        cVertex3D(sPlyColoredOrientedVertex64 aV);
        cVertex3D(sPlyPix4D aV);
        Pt3dr & P() {return mP;}
        Pt3dr & Color() {return mColor;}
    private :
        double mX;
        double mY;
        double mZ;
        double mR;
        double mG;
        double mB;
        Pt3dr mP;
        Pt3dr mColor;
};

cVertex3D::cVertex3D(sPlyColoredOrientedVertex64 aV) :
    mX (double(aV.x)),
    mY (double(aV.y)),
    mZ (double(aV.z)),
    mR (static_cast<double>(aV.red)),
    mG (static_cast<double>(aV.green)),
    mB (static_cast<double>(aV.blue)),
    mP (Pt3dr(mX, mY, mZ)),
    mColor (Pt3dr(mR, mG, mB))
{}

cVertex3D::cVertex3D(sPlyPix4D aV) :
    mX (double(aV.x)),
    mY (double(aV.y)),
    mZ (double(aV.z)),
    mR (static_cast<double>(aV.red)),
    mG (static_cast<double>(aV.green)),
    mB (static_cast<double>(aV.blue)),
    mP (Pt3dr(mX, mY, mZ)),
    mColor (Pt3dr(mR, mG, mB))
{}

cVertex3D::cVertex3D(double & x, double & y, double & z, double & R, double & G, double & B) :
    mX (x),
    mY (y),
    mZ (z),
    mR (R),
    mG (G),
    mB (B),
    mP (Pt3dr(mX, mY, mZ)),
    mColor (Pt3dr(mR, mG, mB))
{
}


class cAppliOrthoDirectFromDenseCloud
{
    public :
        cAppliOrthoDirectFromDenseCloud();
        PlyFile * ReadPly(string & aPlyName);
        bool UpdateOrtho(Pt2dr aPt, Pt3dr aColor, double Z);
        bool UpdateZBuf(Pt2dr aPt, Pt3dr aColor, double Z)   ;
        void Rectify();
        double Equilize(double aVal, double contrast, double luminosityShift);
        void WriteIm2D1chan(tImOrtho & aIm, string aName);
        void WriteIm2D3chan(tImOrthoUINT & aImR, tImOrthoUINT & aImG, tImOrthoUINT & aImB, string aName);
        vector<cVertex3D * > & VVertex3D() {return mVVertex3D;}
        PlyFile * Ply() {return mPlyFile;}
        Pt3dr & PtMin() {return mPtMin;}
        Pt3dr & PtMax() {return mPtMax;}
        double & GSD(){return mGSD;}
        string & NameOrthoOut(){return mNameOrthoOut;}
        bool & IsInverse() {return mIsInverse;}
        bool & DTM() {return mDTM;}
        void ImportOriFolderAndComputeGSD(string & aFolderName);
        double & GSDNominal(){return mGSDNominal;}
        void writeTFW(std::string aImName, double aGSD, Pt2dr offset)  ;
        Pt2dr & OffsetTFW(){return mOffsetTFW;}
        void SetInterpoleMethod(int aTypeCode);
        tInterpolFillHole * Interpolator(){return mInterpole;}
    private:
        double Conv1Cell(tImOrthoUINT &aImgIn, Im2D_REAL8 & aKer, Pt2di & aPos, Pt2di & aSzKer, double & aSomker);
        void Convol_Withker(tImOrthoUINT & aImgIn, Im2D_REAL8 & aKer, Im2D_Bits<1> &aIsPxlSet);
        double Conv1Cell_OnlySetPxl(tImOrthoUINT & aImgIn, Im2D_REAL8 & aKer, Pt2di & aPos, Pt2di & aSzKer, double & aSomker, Im2D_Bits<1> & aIsPxlSet);
        void DoInterpole();
        double ComputeGSDOneIm(CamStenope *aCam);
        vector<cVertex3D * > mVVertex3D;
        PlyFile * mPlyFile;
        tImOrtho mImOrtho;
        tTImOrtho mTImOrtho;
        Pt3dr mPtMin;
        Pt3dr mPtMax;
        double mGSD;
        ElAffin2D mTransTerImg;
        double mContrast;
        double mLumino;
        tImOrthoUINT mImOrthoR;
        tImOrthoUINT mImOrthoG;
        tImOrthoUINT mImOrthoB;
        string mNameOrthoOut;
        bool mIsInverse;
        bool mDTM;
        double mGSDNominal;
        Pt2dr mOffsetTFW;
        tInterpolFillHole * mInterpole;
        Im2D_Bits<1> mIsPxlSet;
        tImOrtho mRImOrthoR;
        tImOrtho mRImOrthoG; // must have Im2D type double for interpolation with ELISE
        tImOrtho mRImOrthoB;
};

cAppliOrthoDirectFromDenseCloud::cAppliOrthoDirectFromDenseCloud():
    mImOrtho      (1, 1, double(Z_DEFAULT)),
    mTImOrtho     (mImOrtho),
    mGSD          (0.2),
    mTransTerImg  (ElAffin2D::Id()),
    mContrast     (1),
    mLumino       (0),
    mImOrthoR     (1,1,int(0)),
    mImOrthoG     (1,1,int(0)),
    mImOrthoB     (1,1,int(0)),
    mNameOrthoOut ("OrthoFromDenseCloud.tif"),
    mIsInverse    (false),
    mGSDNominal   (0.0),
    mOffsetTFW     (Pt2dr(0.0,0.0)),
    mIsPxlSet (1,1),
    mRImOrthoR     (1,1,0.0),
    mRImOrthoG     (1,1,0.0),
    mRImOrthoB     (1,1,0.0)
{}

void cAppliOrthoDirectFromDenseCloud::SetInterpoleMethod(int aTypeCode)
{
    if (aTypeCode == -1)
    {
        mInterpole = NULL;
        return;
    }
    if (aTypeCode == 1) // (def=NONE, 1=bicubic 2=bilinear 3=sinc)
    {
        cCubicInterpKernel * aBic = new cCubicInterpKernel(-0.5);
        // mInterpolBicub = new cTplCIKTabul<tElTiepTri,tElTiepTri>(10,8,-0.5);
        mInterpole = new cTabIM2D_FromIm2D<tPxl>(aBic,1000,false);
        cout<<"Interpolator cCubicInterpKernel set !"<<endl;
    }
    if (aTypeCode == 2) // (def=NONE, 1=bicubic 2=bilinear 3=sinc)
    {
        mInterpole = new cInterpolBilineaire<tPxl>;
        cout<<"Interpolator cInterpolBilineaire set ! Sz Kernel : "<<mInterpole->SzKernel()<<endl;
    }
    if (aTypeCode == 3) // (def=NONE, 1=bicubic 2=bilinear 3=sinc)
    {
        cSinCardApodInterpol1D * aSinC = new cSinCardApodInterpol1D(cSinCardApodInterpol1D::eTukeyApod,5.0,5.0,1e-4,false);
        mInterpole = new cTabIM2D_FromIm2D<tPxl>(aSinC,1000,false);
        cout<<"Interpolator cSinCardApodInterpol1D set !"<<endl;
    }
}

// Compute GSD, thanks ImProjection code
double cAppliOrthoDirectFromDenseCloud::ComputeGSDOneIm(CamStenope * aCamOneIm)
{
    // (tan(resolAngulaire) = ResolSol/H = reslAngulaire : parce que reslAngulaire est petit)
    double aGSDInit = aCamOneIm->ResolutionAngulaire() * aCamOneIm->GetProfondeur();
    return aGSDInit;
}

// Write TFW, thanks DIDRO Code
void cAppliOrthoDirectFromDenseCloud::writeTFW(std::string aImName, double aGSD, Pt2dr offset)
{
    std::string aNameTFW=aImName.substr(0, aImName.size()-3)+"tfw"; // potentiel error if Image have extension more than 3 caracters
    std::ofstream aTFW(aNameTFW.c_str());
    aTFW.precision(12);
    aTFW << aGSD << "\n" << 0 << "\n";
    aTFW << 0 << "\n" <<  -aGSD << "\n";
    aTFW << offset.x << "\n" << offset.y << "\n";
    aTFW.close();
}


void cAppliOrthoDirectFromDenseCloud::ImportOriFolderAndComputeGSD(string & aFolderName)
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
        double aGSD = ComputeGSDOneIm(aCam);
        mGSDNominal += aGSD;
        //cout<<"Im : "<<aNameIm<<" "<<aGSD<<endl;
    }
    mGSDNominal = mGSDNominal/aSetOri.size();
    cout<<"GSD Nominal = "<<mGSDNominal<<endl;
}

double cAppliOrthoDirectFromDenseCloud::Equilize(double aVal, double contrast, double luminosityShift)
{
    return (aVal + luminosityShift)*contrast;
}

PlyFile * cAppliOrthoDirectFromDenseCloud::ReadPly(string & aPlyName)
{
    // Read and give a notion about terrain ground size
    int aMaxX(-1), aMinX(-1), aMaxY(-1), aMinY(-1), aMinZ(-1), aMaxZ(-1);
    int cnt = 0;

    cout<<" ++ Import Ply : "<<aPlyName<<endl;
    PlyFile * ply;               /* description of PLY file */
    //------------------
    int nelems;
    char **elist;
    int file_type;
    float version;
    int nprops;
    int num_elems;
    char *elem_name;

    //read file ply header, give information about version, type element, n° element
    char *pathPlyFile = new char[aPlyName.length() + 1];
    strcpy( pathPlyFile, aPlyName.c_str() );
    ply = ply_open_for_reading(pathPlyFile, &nelems, &elist, &file_type, &version);
    //cout<<"Object has "<<ply->nelems<<" elements "<<" , type = "<<file_type<<" ,version = "<<version<<endl;;
    for (int i = 0; i < ply->nelems; i++)
    {   //scan through each element type
        elem_name = ply->elems[i]->name;
        ply_get_element_description (ply, elem_name , &num_elems, &nprops);  /*get properties of each element type*/

        // check which type is it

        switch(nprops)
        {

        case 6: // x y z r g b
        {
        if (equal_strings ("vertex", elem_name))
        {   //Vertex element type

            for(int j=0;j<nprops;++j)
            {
                ply_get_property(ply, elem_name, &props_Pix4D[j]);
            }
            for (int j = 0; j < num_elems; j++)
            {   //scan all items of this element
                sPlyPix4D aV;
                ply_get_element (ply, (void *) &aV);
                cVertex3D * aVertex = new cVertex3D(aV);
                mVVertex3D.push_back(aVertex);
                if (cnt == 0) //first vertex
                {
                    aMaxX = aVertex->P().x;
                    aMinX = aVertex->P().x;
                    aMaxY = aVertex->P().y;
                    aMinY = aVertex->P().y;
                    aMinZ = aVertex->P().z;
                    aMaxZ = aVertex->P().z;
                }
                else
                {
                    if (aVertex->P().x > aMaxX)
                        aMaxX = aVertex->P().x;
                    if (aVertex->P().x < aMinX)
                        aMinX = aVertex->P().x;
                    if (aVertex->P().y > aMaxY)
                        aMaxY = aVertex->P().y;
                    if (aVertex->P().y < aMinY)
                        aMinY = aVertex->P().y;
                    if (aVertex->P().z > aMaxZ)
                        aMaxZ = aVertex->P().z;
                    if (aVertex->P().z < aMinZ)
                        aMinZ = aVertex->P().z;
                }
                cnt++;
            }
        }
            break;
        }



            case 9: // x y z r g b nx ny nz
            {

            if (equal_strings ("vertex", elem_name))
            {   //Vertex element type

                for(int j=0;j<nprops;++j)
                {
                    ply_get_property(ply, elem_name, &colored_oriented_vert_props[j]);
                }
                for (int j = 0; j < num_elems; j++)
                {   //scan all items of this element
                    sPlyColoredOrientedVertex64 aV;
                    ply_get_element (ply, (void *) &aV);
                    cVertex3D * aVertex = new cVertex3D(aV);
                    mVVertex3D.push_back(aVertex);

                    if (cnt == 0) //first vertex
                    {
                        aMaxX = aVertex->P().x;
                        aMinX = aVertex->P().x;
                        aMaxY = aVertex->P().y;
                        aMinY = aVertex->P().y;
                        aMinZ = aVertex->P().z;
                        aMaxZ = aVertex->P().z;
                    }
                    else
                    {
                        if (aVertex->P().x > aMaxX)
                            aMaxX = aVertex->P().x;
                        if (aVertex->P().x < aMinX)
                            aMinX = aVertex->P().x;
                        if (aVertex->P().y > aMaxY)
                            aMaxY = aVertex->P().y;
                        if (aVertex->P().y < aMinY)
                            aMinY = aVertex->P().y;
                        if (aVertex->P().z > aMaxZ)
                            aMaxZ = aVertex->P().z;
                        if (aVertex->P().z < aMinZ)
                            aMinZ = aVertex->P().z;
                    }
                    cnt++;
                }
            }
                break;
            }
        }

    }
    fclose (ply->fp);
    mPlyFile = ply;
    mPtMin = Pt3dr(aMinX, aMinY, aMinZ);
    mPtMax = Pt3dr(aMaxX, aMaxY, aMaxZ);
    mOffsetTFW = Pt2dr(aMinX, aMaxY);
    // Init trans affine
    /*
    ElAffin2D aTrans (
                       Pt2dr(-aMinX / mGSD, -aMaxY / mGSD), // trans
                       Pt2dr(1/mGSD, 0),                    // X
                       Pt2dr(0, 1/mGSD)                     // Y
                     );
    mTransTerImg = aTrans;

    Pt2dr aSzTer(Pt2dr(mPtMax.x-mPtMin.x , mPtMax.y-mPtMin.y));
    Pt2dr aSzImg(mTransTerImg(Pt2dr(aMaxX, aMaxY)) - mTransTerImg(Pt2dr(aMinX, aMinY)));
    */
    // Test calcul Boxterrain

    // Init trans affine
    ElAffin2D aTrans (
                       Pt2dr(-aMinX / mGSD, aMaxY / mGSD), // trans
                       Pt2dr(1/mGSD, 0),                    // X
                       Pt2dr(0, -1/mGSD)                     // Y
                     );
    mTransTerImg = aTrans;

    Pt2dr aSzTer(Pt2dr(mPtMax.x-mPtMin.x , mPtMax.y-mPtMin.y));
    Pt2dr aSzImg(mTransTerImg(Pt2dr(aMaxX, aMinY)) - mTransTerImg(Pt2dr(aMinX, aMaxY)));



    cout.precision(12);
    cout<<mTransTerImg(Pt2dr(aMinX, aMinY))<<mTransTerImg(Pt2dr(aMaxX, aMaxY))<<endl;



    cout<<"Pt Min : "<<mPtMin<<" -PtMax : "<<mPtMax<<" Sz Terrain : "<<aSzTer<<" Total : "<<cnt<<endl;
    cout<<"SzImg : "<<aSzImg<<endl;
    mImOrtho.Resize(Pt2di(aSzImg));
    mImOrthoR.Resize(Pt2di(aSzImg));
    mImOrthoG.Resize(Pt2di(aSzImg));
    mImOrthoB.Resize(Pt2di(aSzImg));
    //mImOrtho.AugmentSizeTo(Pt2di(aSzImg), double(Z_DEFAULT));

    // calcul equilization parameters
    mContrast = 255/(mPtMax.z  - mPtMin.z);
    mLumino = -mPtMin.z;

    // set interpole label and allocate Im2D<double> for interpolation
    if (mInterpole != NULL)
    {
        mIsPxlSet = Im2D_Bits<1>(aSzImg.x, aSzImg.y, 0);
        mRImOrthoR.Resize(Pt2di(aSzImg));
        mRImOrthoG.Resize(Pt2di(aSzImg));
        mRImOrthoB.Resize(Pt2di(aSzImg));
    }
    //mIsPxlSet = Im2D_Bits(aSzImg.x, aSzImg.y , 0);

    return ply;
}

void cAppliOrthoDirectFromDenseCloud::WriteIm2D1chan(tImOrtho & aIm, string aName)
{
    ELISE_COPY
            (
                aIm.all_pts(),
                aIm.in_proj(),
                Tiff_Im(
                    aName.c_str(),
                    aIm.sz(),
                    GenIm::real8,
                    Tiff_Im::No_Compr,
                    Tiff_Im::BlackIsZero
                    ).out()
                );
}



void cAppliOrthoDirectFromDenseCloud::WriteIm2D3chan(tImOrthoUINT & aImR, tImOrthoUINT & aImG, tImOrthoUINT & aImB, string aName)
{

    Tiff_Im  aTOut
    (
        aName.c_str(),
        aImR.sz(),
        GenIm::u_int1,
        Tiff_Im::No_Compr,
        Tiff_Im::RGB
    );

    ELISE_COPY
    (
        aTOut.all_pts(),
        Virgule(aImR.in(),aImG.in(),aImB.in()),
        aTOut.out()
    );
}

bool cAppliOrthoDirectFromDenseCloud::UpdateZBuf(Pt2dr aPt, Pt3dr aColor, double Z)
{
    // Check if point is outside image
    Pt2di aPtInt = Pt2di(aPt);
    if (mImOrtho.Inside(aPtInt))
    {
        double valZ = mImOrtho.GetR(aPtInt);
        if (valZ == Z_DEFAULT)
        {
            mImOrtho.SetR(aPtInt , Equilize(Z, mContrast, mLumino));
            return true;
        }
        else
        {

                if (valZ > Z)
                {
                    mImOrtho.SetR(aPtInt , Equilize(Z, mContrast, mLumino));
                    return true;
                }
                else
                    return false;

        }
    }
    return false;
}

bool cAppliOrthoDirectFromDenseCloud::UpdateOrtho(Pt2dr aPt, Pt3dr aColor, double Z)
{
    // Check if point is outside image
    Pt2di aPtInt = Pt2di(aPt);      // punaise, tu perds la precision ici !
    if (mImOrtho.Inside(aPtInt))
    {
        double valZ = mImOrtho.GetR(aPtInt);
        if (valZ == Z_DEFAULT)
        {
            mImOrtho.SetR(aPtInt , Z);
            mImOrthoR.SetR(aPtInt, aColor.x);
            mImOrthoG.SetR(aPtInt, aColor.y);
            mImOrthoB.SetR(aPtInt, aColor.z);
            if (mInterpole != NULL)
            {
                mIsPxlSet.SetI(aPtInt, 1);
                mRImOrthoR.SetR(aPtInt, aColor.x);
                mRImOrthoG.SetR(aPtInt, aColor.y);
                mRImOrthoB.SetR(aPtInt, aColor.z);
            }
            return true;
        }
        else
        {
            if (!mIsInverse)
            {
                if (valZ > Z) //attention : Pix4D code differently (<)
                {
                    mImOrtho.SetR(aPtInt , Z);
                    mImOrthoR.SetR(aPtInt, aColor.x);
                    mImOrthoG.SetR(aPtInt, aColor.y);
                    mImOrthoB.SetR(aPtInt, aColor.z);
                    if (mInterpole != NULL)
                    {
                        mIsPxlSet.SetI(aPtInt, 1);
                        mRImOrthoR.SetR(aPtInt, aColor.x);
                        mRImOrthoG.SetR(aPtInt, aColor.y);
                        mRImOrthoB.SetR(aPtInt, aColor.z);
                    }
                    return true;
                }
                else
                    return false;
            }
            else
            {
                if (valZ < Z) //attention : Pix4D code differently (<)
                {
                    mImOrtho.SetR(aPtInt , Z);
                    mImOrthoR.SetR(aPtInt, aColor.x);
                    mImOrthoG.SetR(aPtInt, aColor.y);
                    mImOrthoB.SetR(aPtInt, aColor.z);
                    if (mInterpole != NULL)
                    {
                        mIsPxlSet.SetI(aPtInt, 1);
                        mRImOrthoR.SetR(aPtInt, aColor.x);
                        mRImOrthoG.SetR(aPtInt, aColor.y);
                        mRImOrthoB.SetR(aPtInt, aColor.z);
                    }
                    return true;
                }
                else
                    return false;
            }
        }
    }
    return false;
}

void cAppliOrthoDirectFromDenseCloud::Rectify()
{
    cout<<"CONFIRM TRANSF : "<<mTransTerImg.I00()<<mTransTerImg.I01()<<mTransTerImg.I10()<<endl;

    cout<<"Rectifying image..."<<endl;

    string aMess = (mInterpole == NULL) ? " NO " : " YES ";

    cout<<"Interpole : "<<aMess<<endl;

    for (uint i=0; i<mVVertex3D.size(); i++)
    {
        cVertex3D * aVertex = mVVertex3D[i];
        Pt2dr aP2DTer(aVertex->P().x, aVertex->P().y);
        double Z = aVertex->P().z;
        Pt2dr aP2DImg = mTransTerImg(aP2DTer);
        //cout<<aP2DTer<<aP2DImg<<endl;
        /*bool isUpdateZ = */ UpdateOrtho(aP2DImg, aVertex->Color(), Z); // Warn 
    }
    cout<<"Write ortho direct image : "<<mNameOrthoOut<<endl;
    WriteIm2D3chan(mImOrthoR, mImOrthoG, mImOrthoB, mNameOrthoOut);
    writeTFW(mNameOrthoOut, mGSD, mOffsetTFW);
    if (mDTM)
    {
        cout<<"Write DTM image : "<<mNameOrthoOut + "_DTM.tif"<<endl;
        WriteIm2D1chan(mImOrtho, mNameOrthoOut + "_DTM.tif");
    }
    if (mInterpole != NULL)
    {
        DoInterpole();
    }
}

double cAppliOrthoDirectFromDenseCloud::Conv1Cell(tImOrthoUINT & aImgIn, Im2D_REAL8 & aKer, Pt2di & aPos, Pt2di & aSzKer, double & aSomker)
{
    double aSom=0;
    for (int aKx=-aSzKer.x; aKx<=aSzKer.x; aKx++)
    {
        for (int aKy=-aSzKer.y; aKy<=aSzKer.y; aKy++)
        {
            Pt2di aVois(aKx, aKy);
            aSom += aImgIn.GetI(aPos + aVois) * aKer.GetI(aVois + aSzKer);
            //cout<<"Img "<<(aPos + aVois)<<aImgIn.GetI(aPos + aVois)<<" -aKer "<<(aVois + aSzKer)<<aKer.GetI(aVois + aSzKer)<<endl;
        }
    }
    return (aSom/aSomker);
}

double cAppliOrthoDirectFromDenseCloud::Conv1Cell_OnlySetPxl(tImOrthoUINT & aImgIn, Im2D_REAL8 & aKer, Pt2di & aPos, Pt2di & aSzKer, double & aSomker, Im2D_Bits<1> & aIsPxlSet)
{
    double aSom=0;
    double aSomKerUpdate = 0;
    for (int aKx=-aSzKer.x; aKx<=aSzKer.x; aKx++)
    {
        for (int aKy=-aSzKer.y; aKy<=aSzKer.y; aKy++)
        {
            Pt2di aVois(aKx, aKy);
            if (aIsPxlSet.GetI(aPos + aVois) != 0)
            {
                aSom += aImgIn.GetI(aPos + aVois) * aKer.GetI(aVois + aSzKer);
                aSomKerUpdate += aKer.GetI(aVois + aSzKer);
            }
            //cout<<"Img "<<(aPos + aVois)<<aImgIn.GetI(aPos + aVois)<<" -aKer "<<(aVois + aSzKer)<<aKer.GetI(aVois + aSzKer)<<endl;
        }
    }
    return (aSom/aSomKerUpdate);
}

void cAppliOrthoDirectFromDenseCloud::Convol_Withker(tImOrthoUINT & aImgIn, Im2D_REAL8 & aKer, Im2D_Bits<1> & aIsPxlSet)
{
    Pt2di aSzKer(round_up((aKer.sz().x-1)/2), round_up((aKer.sz().y-1)/2));
    Pt2di aRun;

    double aSomKer = aKer.som_rect();
    if (aSomKer == 0)
        aSomKer = 1;
    int aCnt = 0;
    int aTotal = 0;

    for (aRun.x = aSzKer.x ;aRun.x < aImgIn.sz().x-aSzKer.x; aRun.x++)
    {
        for (aRun.y = aSzKer.y ;aRun.y < aImgIn.sz().y-aSzKer.y; aRun.y++)
        {
            if (aIsPxlSet.GetI(aRun) == 0)
            {
                //double aRes = Conv1Cell(aImgIn, aKer, aRun, aSzKer, aSomKer);
                aTotal++;
                double aRes = Conv1Cell_OnlySetPxl(aImgIn, aKer, aRun, aSzKer, aSomKer, aIsPxlSet);
                if (aRes > 0)
                {
                    aCnt++;
                    aImgIn.SetI_SVP(aRun, aRes);
                }
            }
        }
    }
    cout<<"Finish Interpole, Nb Pixel Interpole = "<<aCnt<<"/"<<aTotal<<" total, for each of 4 image channels (R, G, B, DTM)"<<endl;
}


void cAppliOrthoDirectFromDenseCloud::DoInterpole()
{
    if (mInterpole == NULL)
        return;
    else
    {
        cout<<"Interpolator initialized ! Do interpol..."<<endl;
        cout<<"Im Label Sz : "<<mIsPxlSet.sz()<<endl;
        string aName = mNameOrthoOut + "Label.tif";
        ELISE_COPY
                (
                    mIsPxlSet.all_pts(),
                    mIsPxlSet.in_proj(),
                    Tiff_Im(
                        aName.c_str(),
                        mIsPxlSet.sz(),
                        //GenIm::int1,
                        mIsPxlSet.TypeEl(),
                        Tiff_Im::No_Compr,
                        Tiff_Im::BlackIsZero
                        ).out()
                 );

        Im2D_REAL8 aKernel(5,5,
                            "0.2 0.3 1.2 0.3 0.2 "
                            "0.2 0.3 0.8 0.3 0.2 "
                            "0.5 0.5 0 0.5 0.5 "
                            "0.2 0.3 0.8 0.3 0.2 "
                            " 0.2 0.3 1.2 0.3 0.2"
                            );



        // Interpol
        Convol_Withker(mImOrthoB, aKernel, mIsPxlSet);
        Convol_Withker(mImOrthoG, aKernel, mIsPxlSet);
        Convol_Withker(mImOrthoR, aKernel, mIsPxlSet);
        WriteIm2D3chan(mImOrthoR,mImOrthoG,mImOrthoB, mNameOrthoOut + "_FillHole.tif");
    }
}

std::string BannerOrthoDirectFromDenseCloud()
{
    std::string banniere = "\n";
    banniere += "************************************************************************* \n";
    banniere += "**                                                                     ** \n";
    banniere += "************************************************************************* \n";
    return banniere;
}

int OrthoDirectFromDenseCloud_main(int argc,char ** argv)
{

    cout<<BannerOrthoDirectFromDenseCloud();
    string aPlyFileName;
    string aOrthoOut = "OrthoFromDenseCloud.tif";
    double aGSD(0.2);
    bool aIsInverse = false;
    bool aDTM = true;
    string aOri;
    int aInterpole = -1; // (-1=NONE, 1=bicubic 2=bilinear 3=sinc)


    ElInitArgMain
    (
          argc, argv,
          LArgMain()
                << EAMC(aPlyFileName, "Ply file for 3D point cloud", eSAM_IsExistFile),
          LArgMain()
                << EAM(aGSD, "GSD",false, "GSD (m/pixel) - def = 0.2")
                << EAM(aOrthoOut, "Out",false, "Ortho output filename (def = OrthoFromDenseCloud.tif")
                << EAM(aIsInverse, "Inv",false, "def = false => highest Z to update ZBuffer")
                << EAM(aDTM, "DTM",false, "def = true => export Z Buffer image")
                << EAM(aOri, "Ori",false, "Ori folder - compute and rectify ortho with GSD nominal")
                << EAM(aInterpole, "Int",false, "Interpolation method to fill lacked pixel in orthophoto - (def=NONE, 1=bicubic 2=bilinear 3=sinc)")
    );

 cAppliOrthoDirectFromDenseCloud * aAppli = new cAppliOrthoDirectFromDenseCloud();
 aAppli->IsInverse() = aIsInverse;

 if (EAMIsInit(&aOri))
 {
     aAppli->ImportOriFolderAndComputeGSD(aOri);
     aAppli->GSD() = aAppli->GSDNominal();
     cout<<"Orthophoto will be rectify as GSD nominal = "<<aAppli->GSD()<<endl;
 }
 else
 {
      aAppli->GSD() = aGSD;
      cout<<"Orthophoto will be rectify as GSD = "<<aAppli->GSD()<<endl;
 }

 aAppli->SetInterpoleMethod(aInterpole);

 aAppli->ReadPly(aPlyFileName);
 aAppli->NameOrthoOut() = aOrthoOut;
 aAppli->DTM() = aDTM;
 aAppli->Rectify();

 cout<<endl<<endl<<"********  Finish  **********"<<endl;
 return EXIT_SUCCESS;
}
