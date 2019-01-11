#include "orthodirectfromdensecloud.h"

class cAppliOrthoDirectFromDenseCloud;
// rectification directly point cloud to plan image

typedef Im2D<double,double>  tImOrtho;
typedef TIm2D<double,double> tTImOrtho;
typedef Im2D_U_INT1 tImOrthoUINT;
#define Z_DEFAULT -10000.0

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
        Pt2dr & GSD(){return mGSD;}
        string & NameOrthoOut(){return mNameOrthoOut;}
        bool & IsInverse() {return mIsInverse;}
    private:
        vector<cVertex3D * > mVVertex3D;
        PlyFile * mPlyFile;
        tImOrtho mImOrtho;
        tTImOrtho mTImOrtho;
        Pt3dr mPtMin;
        Pt3dr mPtMax;
        Pt2dr mGSD;
        ElAffin2D mTransTerImg;
        double mContrast;
        double mLumino;
        tImOrthoUINT mImOrthoR;
        tImOrthoUINT mImOrthoG;
        tImOrthoUINT mImOrthoB;
        string mNameOrthoOut;
        bool mIsInverse;
};

cAppliOrthoDirectFromDenseCloud::cAppliOrthoDirectFromDenseCloud():
    mImOrtho      (1, 1, double(Z_DEFAULT)),
    mTImOrtho     (mImOrtho),
    mGSD          (Pt2dr(0.2, 0.2)),
    mTransTerImg  (ElAffin2D::Id()),
    mContrast     (1),
    mLumino       (0),
    mImOrthoR     (1,1,int(0)),
    mImOrthoG     (1,1,int(0)),
    mImOrthoB     (1,1,int(0)),
    mNameOrthoOut ("OrthoFromDenseCloud.tif"),
    mIsInverse    (false)
{
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

    // Init trans affine
    ElAffin2D aTrans (
                       Pt2dr(-aMinX / mGSD.x, -aMinY / mGSD.y),
                       Pt2dr(1/mGSD.x, 0),
                       Pt2dr(0, 1/mGSD.y)
                     );
    mTransTerImg = aTrans;

    Pt2dr aSzTer(Pt2dr(mPtMax.x-mPtMin.x , mPtMax.y-mPtMin.y));
    Pt2dr aSzImg(mTransTerImg(Pt2dr(aMaxX, aMaxY)) - mTransTerImg(Pt2dr(aMinX, aMinY)));

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
    Pt2di aPtInt = Pt2di(aPt);
    if (mImOrtho.Inside(aPtInt))
    {
        double valZ = mImOrtho.GetR(aPtInt);
        if (valZ == Z_DEFAULT)
        {
            mImOrtho.SetR(aPtInt , Z);
            mImOrthoR.SetR(aPtInt, aColor.x);
            mImOrthoG.SetR(aPtInt, aColor.y);
            mImOrthoB.SetR(aPtInt, aColor.z);
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
    for (uint i=0; i<mVVertex3D.size(); i++)
    {
        cVertex3D * aVertex = mVVertex3D[i];
        Pt2dr aP2DTer(aVertex->P().x, aVertex->P().y);
        double Z = aVertex->P().z;
        Pt2dr aP2DImg = mTransTerImg(aP2DTer);
        bool isUpdateZ = UpdateOrtho(aP2DImg, aVertex->Color(), Z);
    }
    WriteIm2D3chan(mImOrthoR, mImOrthoG, mImOrthoB, mNameOrthoOut);
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
    Pt2dr aGSD(0.2,  0.2);
    bool aIsInverse = false;


    ElInitArgMain
    (
          argc, argv,
          LArgMain()
                << EAMC(aPlyFileName, "Ply file for 3D point cloud", eSAM_IsExistFile),
          LArgMain()
                << EAM(aGSD, "GSD",false, "GSD (m/pixel)")
                << EAM(aOrthoOut, "Out",false, "Ortho output filename (def = OrthoFromDenseCloud.tif")
                << EAM(aIsInverse, "Inv",false, "def = false => highest Z to update ZBuffer")
    );

 cAppliOrthoDirectFromDenseCloud * aAppli = new cAppliOrthoDirectFromDenseCloud();
 aAppli->IsInverse() = aIsInverse;
 aAppli->GSD() = aGSD;
 aAppli->ReadPly(aPlyFileName);
 aAppli->NameOrthoOut() = aOrthoOut;
 aAppli->Rectify();
 cout<<endl<<endl<<"********  Finish  **********"<<endl;
 return EXIT_SUCCESS;
}
