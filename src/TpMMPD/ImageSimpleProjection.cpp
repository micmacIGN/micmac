/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

#include "imagesimpleprojection.h"

/********************************************************************/
/*                                                                  */
/*         cISR_ColorImg                                                 */
/*                                                                  */
/********************************************************************/


cISR_ColorImg::cISR_ColorImg(std::string filename) :
    mImgName(filename)
{
    Tiff_Im mTiffImg= Tiff_Im::UnivConvStd(mImgName);
    mImgSz.x=mTiffImg.sz().x;
    mImgSz.y=mTiffImg.sz().y;
    mImgR=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgG=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgB=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgRT=new TIm2D<U_INT1,INT4>(*mImgR);
    mImgGT=new TIm2D<U_INT1,INT4>(*mImgG);
    mImgBT=new TIm2D<U_INT1,INT4>(*mImgB);
    ELISE_COPY(mImgR->all_pts(),mTiffImg.in(),Virgule(mImgR->out(),mImgG->out(),mImgB->out()));
}


cISR_ColorImg::cISR_ColorImg(Pt2di sz) :
    mImgName(""),
    mImgSz(sz)
{
    mImgR=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgG=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgB=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgRT=new TIm2D<U_INT1,INT4>(*mImgR);
    mImgGT=new TIm2D<U_INT1,INT4>(*mImgG);
    mImgBT=new TIm2D<U_INT1,INT4>(*mImgB);
}

cISR_ColorImg::~cISR_ColorImg()
{
    delete mImgR;
    delete mImgG;
    delete mImgB;
    delete mImgRT;
    delete mImgGT;
    delete mImgBT;
} 

cISR_Color cISR_ColorImg::get(Pt2di pt) // the method get() return an objet "color" point
{
    return cISR_Color(mImgRT->get(pt,0),mImgGT->get(pt,0),mImgBT->get(pt,0));
}

cISR_Color cISR_ColorImg::getr(Pt2dr pt)
{
    // get (pt, 0) est plus robuste que get (pt), retourne 0 si le point est hors images
    return cISR_Color(mImgRT->getr(pt,0),mImgGT->getr(pt,0),mImgBT->getr(pt,0));
}

void cISR_ColorImg::set(Pt2di pt, cISR_Color color)
{
    U_INT1 ** aImRData=mImgR->data();
    U_INT1 ** aImGData=mImgG->data();
    U_INT1 ** aImBData=mImgB->data();
    aImRData[pt.y][pt.x]=color.r();
    aImGData[pt.y][pt.x]=color.g();
    aImBData[pt.y][pt.x]=color.b();
}

void cISR_ColorImg::write(std::string filename)
{
    ELISE_COPY
            (
                mImgR->all_pts(),
                Virgule( mImgR->in(), mImgG->in(), mImgB->in()) ,
                Tiff_Im(
                    filename.c_str(),
                    mImgSz,
                    GenIm::u_int1,
                    Tiff_Im::No_Compr,
                    Tiff_Im::RGB,
                    Tiff_Im::Empty_ARG ).out()
                );

}

cISR_ColorImg  cISR_ColorImg::ResampleColorImg(double aFact)
{
    Pt2di aSzR = round_up(Pt2dr(mImgSz)/aFact);

    cISR_ColorImg aResampled(aSzR);

    Fonc_Num aFInR = StdFoncChScale
            (
                this->mImgR->in_proj(),
                Pt2dr(0,0),
                Pt2dr(aFact,aFact)
                );
    Fonc_Num aFInG = StdFoncChScale(this->mImgG->in_proj(),Pt2dr(0,0),Pt2dr(aFact,aFact));
    Fonc_Num aFInB = StdFoncChScale(this->mImgB->in_proj(),Pt2dr(0,0),Pt2dr(aFact,aFact));

    ELISE_COPY(aResampled.mImgR->all_pts(),aFInR,aResampled.mImgR->out());
    ELISE_COPY(aResampled.mImgG->all_pts(),aFInG,aResampled.mImgG->out());
    ELISE_COPY(aResampled.mImgB->all_pts(),aFInB,aResampled.mImgB->out());
    return aResampled;
}

/********************************************************************/
/*                                                                  */
/*         cISR_Ima                                                 */
/*                                                                  */
/********************************************************************/

// constructor of class ISR Image
cISR_Ima::cISR_Ima(cISR_Appli & anAppli,const std::string & aName,int aAlti,int aDZ, std::string & aPrefix,bool aQuick) :
    mAlti   (aAlti),
    mZTerrain (0),
    mDeZoom (aDZ),
    mQuickResampling(aQuick),
    //private
    mAppli  (anAppli),
    mName   	(aName), // Jpg image for e.g.
    mNameTiff (NameFileStd(mName,3,false,true,true,true)), // associated tif 3 channel image
    //mTifIm  (Tiff_Im::StdConvGen(mAppli.Dir() + mNameTiff,3,true)),
    mNameOri  (mAppli.NameIm2NameOri(mName)),
    mPrefix   (aPrefix),
    mCam      (CamOrientGenFromFile(mNameOri,mAppli.ICNM())),
    mH	(cElHomographie::Id())
{
}

void cISR_Ima::ApplyImProj()
{
    //  For a given ground surface elevation, compute the rectified image (rectification==redressement)
    Pt2di aSz = this->SzXY();
    cISR_ColorImg ImCol(mNameTiff.c_str());
    cISR_ColorImg ImColRect(aSz);
    Pt2di aP;
    std::cout << "Beginning of rectification for oblique image " << this->mName  << "   -------------- \n";

    // Loop on every column and line of the rectified image
    for (aP.x=0 ; aP.x<aSz.x; aP.x++)
    {
        // compute X coordinate in ground/object geometry
        double aX=mBorder[0]+mLoopGSD * aP.x;
        for (aP.y=0 ; aP.y<aSz.y; aP.y++)
        {
            // compute Y coordinate in ground/object geometry
            double aY=mBorder[3]-mLoopGSD * aP.y;
            // define the point position in ground geometry
            Pt3dr aPTer(aX,aY,mZTerrain);
            // project this point in the initial image
            Pt2dr aPIm0 = mCam->R3toF2(aPTer);
            //std::cout << " point 2D image  :: " << aPIm0 <<"    \n";
            //std::cout << " point 3D terrain :: " << aPTer <<"    \n";

            // get the radiometric value at this position
            cISR_Color aCol=ImCol.getr(aPIm0);
            // write the value on the rectified image
            ImColRect.set(aP,aCol);
        }
    }

    // write the rectified image in the working directory
    this->WriteImage(ImColRect, "ProR3F2");
    // write the tfw file
    this->GenTFW();
    std::cout << "End of rectification for oblique image " << this->mName  << "  ------------------- \n \n";
}

void cISR_Ima::ApplyImHomography()
{
    //  generat the rectified image by using an homography, faster than with R3toF2
    Pt2di aSz = this->SzXY();
    cISR_ColorImg ImCol(mNameTiff.c_str());
    cISR_ColorImg ImColRect(aSz);
    Pt2di aP;
    std::cout << "Beginning of rectification by homography for oblique image " << this->mName  << "   -------------- \n";

    // Loop on every column and line of the rectified image
    for (aP.x=0 ; aP.x<aSz.x; aP.x++)
    {
        // compute X coordinate in ground/object geometry
        double aX=mBorder[0]+mLoopGSD * aP.x;

        for (aP.y=0 ; aP.y<aSz.y; aP.y++)
        {
            // compute Y coordinate in ground/object geometry
            double aY=mBorder[3]-mLoopGSD * aP.y;
            // define the point planimetric position in ground geometry
            Pt2dr aPTerPlani(aX,aY);
            // project this point in the initial image using the homography relationship
            Pt2dr aPIm0=mH.Direct(aPTerPlani);
            //std::cout << " point 2D image  :: " << aPIm0 <<"    \n";
            //std::cout << " point 3D terrain :: " << aPTerPlani <<"    \n";

                // get the radiometric value at this position
                cISR_Color aCol=ImCol.getr(aPIm0);
                // write the value on the rectified image
                //cout<<aPTerPlani<<aPIm0<<endl;
                ImColRect.set(aP,aCol);

        }
    }

    this->WriteImage(ImColRect, "Rectified");
    // write the tfw file
    this->GenTFW();
    
    std::cout << "End of rectification by homography for oblique image " << this->mName  << "  ------------------- \n \n";
}


void cISR_Ima::GenTFW()
{
    std::string aNameTFW = mPrefix+"-"+this->mName + ".tfw";
    std::ofstream aFtfw(aNameTFW.c_str());
    aFtfw.precision(0);
    aFtfw << mFGSD << "\n" << 0 << "\n";
    aFtfw << 0 << "\n" << -mFGSD << "\n";
    aFtfw << mBorder[0] << "\n" << mBorder[3] << "\n";
    aFtfw.close();
}

void cISR_Ima::GenTFW(double mGSD, Pt2dr offset, string aPrefix)
{
    std::string aNameTFW = aPrefix+"-"+this->mName + ".tfw";
    std::ofstream aFtfw(aNameTFW.c_str());
    aFtfw.precision(0);
    aFtfw << mGSD << "\n" << 0 << "\n";
    aFtfw << 0 << "\n" << mGSD << "\n";
    aFtfw << offset.x << "\n" << offset.y << "\n";
    aFtfw.close();
}

void cISR_Ima::WriteImage(cISR_ColorImg & aImage)
{
    // write the rectified image in the working directory
    std::string aNameImProj= mPrefix+"-"+this->mName+".tif";
    
    // performed a resample of the rectified image.
    if ((mDeZoom!=1) & (mQuickResampling!=1))
    {
        cISR_ColorImg ImResampled = aImage.ResampleColorImg(mDeZoom);
        ImResampled.write(aNameImProj);
        std::cout << "Resampling of rectified image (dezoom factor of " << mDeZoom << ") \n";
    }
    else aImage.write(aNameImProj);
}

void cISR_Ima::WriteImage(cISR_ColorImg & aImage, string aPrefix)
{
    // write the rectified image in the working directory
    std::string aNameImProj= aPrefix+"-"+this->mName+".tif";

    // performed a resample of the rectified image.
    if ((mDeZoom!=1) & (mQuickResampling!=1))
    {
        cISR_ColorImg ImResampled = aImage.ResampleColorImg(mDeZoom);
        ImResampled.write(aNameImProj);
        std::cout << "Resampling of rectified image (dezoom factor of " << mDeZoom << ") \n";
    }
    else aImage.write(aNameImProj);
}

void cISR_Ima::InitGeomTerrain()
{
    // if the user has defined a Flight altitude, we assume the soil elevetion to be at Z=position of the camera-flight altitude.
    // else, the information of camera depth is used instead of flight altitude.
    if (mAlti==0) mAlti=static_cast<int>(mCam->GetProfondeur());
    // get the pseudo optical center of the camera (position XYZ of the optical center)
    Pt3dr OC=mCam->PseudoOpticalCenter();
    mZTerrain=static_cast<int>(OC.z-mAlti);
    // des fois l'info alti est notée dans mCam mais pas l'info Profondeur. c'est peut-etre uniquement le cas pour les mauvais orientation
    //if (mZTerrain<0) (mZTerrain=static_cast<int>(OC.z-mCam->GetAlti()));
    if (mZTerrain<0) {
        std::cout << "For Image  " << this->mName  << " \n";
        ELISE_ASSERT(false,"Ground Surface Elevation is below 0 (check FAlti)."); }
    // declare the 4 3Dpoints used for determining the XYZ coordinates of the 4 corners of the camera
    Pt3dr P1;
    Pt3dr P2;
    Pt3dr P3;
    Pt3dr P4;
    // project the 4 corners of the camera, ground surface assumed to be a plane
    mCam->CoinsProjZ(P1, P2, P3, P4, mZTerrain);
    // determine the ground sample distance.
    mIGSD=std::abs (mCam->ResolutionSol(Pt3dr(OC.x,OC.y,mZTerrain))); //initial ground sample distance
    mFGSD=mIGSD*mDeZoom; // final ground sample distance , different from Initial if dezoom is applied
    mLoopGSD=mIGSD;

    // determine  xmin,xmax,ymin, ymax
    double x[4]={P1.x,P2.x,P3.x,P4.x};
    double y[4]={P1.y,P2.y,P3.y,P4.y};
    double *maxx=std::max_element(x,x+4);
    double *minx=std::min_element(x,x+4);
    double *maxy=std::max_element(y,y+4);
    double *miny=std::min_element(y,y+4);
    //int border[4]={static_cast<int>(*minx),static_cast<int>(*maxx),static_cast<int>(*miny),static_cast<int>(*maxy)};
    mBorder[0]=static_cast<int>(*minx);
    mBorder[1]=static_cast<int>(*maxx);
    mBorder[2]=static_cast<int>(*miny);
    mBorder[3]=static_cast<int>(*maxy);
    // determine the size in pixel of the projected image - without dezoom
    int SzX=(mBorder[1]-mBorder[0])/mLoopGSD;
    int SzY=(mBorder[3]-mBorder[2])/mLoopGSD;

    cout<<"Foot Print Size in pixel = "<<SzX<<" , "<<SzY<<endl;

    mSzImRect = Pt2di(SzX,SzY);


}

void cISR_Ima::Estime4PtsProjectiveTransformation()
{
    //   R3 : coordonné terrain absolue
    //   L3 : coordonné terrain dans repere camera
    //   C2 : coordonné 2D caméra
    //   F2 : coordonné 2D image
    //
    //       Orientation      Projection      Distortion
    //   R3 -------------> L3------------>C2------------->F2

    // determine 4 correspondant 2D-3D from 4 image corner
    cout<<"Estime Projective Transf..."<<endl;
    Pt3dr P1;
    Pt3dr P2;
    Pt3dr P3;
    Pt3dr P4;
    mZTerrain = mCam->PseudoOpticalCenter().z - mCam->GetProfondeur();
    cout<<" + Z = "<<mZTerrain<<endl;
    mCam->CoinsProjZ(P1, P2, P3, P4, mZTerrain);
    vector<Pt3dr> aVP;
    aVP.push_back(P1);
    aVP.push_back(P2);
    aVP.push_back(P3);
    aVP.push_back(P4);

    // determine correspondant 2D, en prenent en compte la distorsion
    vector<Pt2dr> aVp;
    Pt2dr p1 = mCam->R3toF2(P1);
    Pt2dr p2 = mCam->R3toF2(P2);
    Pt2dr p3 = mCam->R3toF2(P3);
    Pt2dr p4 = mCam->R3toF2(P4);
    aVp.push_back(p1);
    aVp.push_back(p2);
    aVp.push_back(p3);
    aVp.push_back(p4);

    cout<<"Correspondant : p_backproj,P_3D,p_theorie "<<endl;
    cout<<p1<<P1<<endl;
    cout<<p2<<P2<<endl;
    cout<<p3<<P3<<endl;
    cout<<p4<<P4<<endl;
    // estimer 8 params de perspective projection (e0 e1 e2 f0 f1 f2 g1 g2)
    L2SysSurResol aSys(8);
    for (uint i=0; i<4; i++)
    {
        Pt2dr ap = aVp[i];
        Pt3dr aP = aVP[i];
        double coeffEQ1[8] = {aP.x*ap.x, -ap.x, 0, aP.x*ap.y, -ap.y, 0, -1, 0};
        double coeffEQ2[8] = {aP.y*ap.x, 0, -ap.x, aP.y*ap.y, 0, -ap.y, 0, -1};
        double obs1 = -aP.x;
        double obs2 = -aP.y;
        aSys.AddEquation(1.0, coeffEQ1, obs1);
        aSys.AddEquation(1.0, coeffEQ2, obs2);
    }
    bool OK = false;
    Im1D_REAL8 aSol(8);
    double * aData;
    aSol = aSys.Solve(&OK);
    vector<double> aParamProj;
    if(OK)
    {
        aData = aSol.data();
        cout<<"Coeff (e0,e1,e2,f0,f1,f2,g1,g2): ";
        for (int i=0; i<8; i++)
        {
            cout<<aData[i]<<" ";
            aParamProj.push_back(aData[i]);
        }
        cout<<endl;
    }
    this->RectifyByProjectiveTransformation(aVp, aVP, aParamProj);
}

void cISR_Ima::RectifyByProjectiveTransformation(vector<Pt2dr> aVp, vector<Pt3dr> aVP, vector<double> aParamProj)
{
    cout<<"Rectify : "<<endl;
    double e0 = aParamProj[0];
    double e1 = aParamProj[1];
    double e2 = aParamProj[2];
    double f0 = aParamProj[3];
    double f1 = aParamProj[4];
    double f2 = aParamProj[5];
    double g1 = aParamProj[6];
    double g2 = aParamProj[7];

        // convertir coins image metrique aux coins image pixel
            // calcul GSD :
            double aGSDInit = mCam->ResolutionAngulaire() * mCam->GetProfondeur(); // (tan(resolAngulaire) = ResolSol/H = reslAngulaire : parce que reslAngulaire est petit)
            cout<<" + GSD : "<<aGSDInit<<endl; // metre/pixel
            // Box sol :
            Pt2dr aBoxSolMin(ElMin4(aVP[0].x,aVP[1].x,aVP[2].x,aVP[3].x), ElMin4(aVP[0].y,aVP[1].y,aVP[2].y,aVP[3].y));
            Pt2dr aBoxSolMax(ElMax4(aVP[0].x,aVP[1].x,aVP[2].x,aVP[3].x), ElMax4(aVP[0].y,aVP[1].y,aVP[2].y,aVP[3].y));
            cout<<"Box sol : "<<aBoxSolMin<<aBoxSolMax<<aBoxSolMax-aBoxSolMin<<endl;
            // Determiner offset georef et taille d'image rectifié :
            Pt2dr offset(aBoxSolMin.x, -aBoxSolMax.y);
            Pt2dr sZImRecMetric(aBoxSolMax-aBoxSolMin);
            Pt2di sZImRecPxl((aBoxSolMax-aBoxSolMin)/aGSDInit);
            cout<<"offset : "<<offset<<" , sZRec : "<<sZImRecPxl<<" , sZOrg : "<<mCam->SzPixel()<<endl;

        // parcourir espace terrain, puis chercher point sur terrain par la projective
        Pt2di apImg(0,0);
        Pt2di aSz(mCam->SzPixel());
        cISR_ColorImg ImCol(mNameTiff.c_str());
        cISR_ColorImg ImColRectInv(aSz);
        Pt3dr offset3D(offset.x, offset.y, 0);
        for (apImg.x=0; apImg.x<aSz.x; apImg.x++)
        {
            for (apImg.y=0; apImg.y<aSz.y; apImg.y++)
            {
                Pt2dr apImgInit;
                Pt3dr aPTer(0,0,mZTerrain);
                aPTer.x = apImg.x*aGSDInit + aBoxSolMin.x;
                aPTer.y = apImg.y*aGSDInit + aBoxSolMin.y;
                apImgInit.x = ((f2-f0*g2)*aPTer.x + (f0*g1-f1)*aPTer.y + (g2*f1-g1*f2)) / ((e2*f0-e0*f2)*aPTer.x + (e0*f1-e1*f0)*aPTer.y + (e1*f2-e2*f1));
                apImgInit.y = ((e0*g2-e2)*aPTer.x + (e1-e0*g1)*aPTer.y + (e2*g1-e1*g2)) / ((e2*f0-e0*f2)*aPTer.x + (e0*f1-e1*f0)*aPTer.y + (e1*f2-e2*f1));
                //cout<<apImg<<aP<<endl;
                if (apImgInit.x>0 && apImgInit.y>0 && apImgInit.x<aSz.x && apImgInit.y<aSz.y)
                {
                    cISR_Color aCol=ImCol.getr(apImgInit);
                    ImColRectInv.set(apImg,aCol);
                }
            }
        }
        this->WriteImage(ImColRectInv, "Proj4Pts");
        // write the tfw file
        this->GenTFW(aGSDInit, offset, "Proj4Pts");
        cout<<"Done"<<endl;
}


void cISR_Ima::RectifyByHomography()
{
    // compute homography between 2 plan
    cout<<"Estime Homography..."<<endl;
    Pt3dr P1;    Pt3dr P2;    Pt3dr P3;    Pt3dr P4;
    mZTerrain = mCam->PseudoOpticalCenter().z - mCam->GetProfondeur();
    cout<<" + Z = "<<mZTerrain<<endl;
    mCam->CoinsProjZ(P1, P2, P3, P4, mZTerrain);
    vector<Pt3dr> aVP;
    aVP.push_back(P1);    aVP.push_back(P2);    aVP.push_back(P3);    aVP.push_back(P4);

    // determine correspondant 2D, en prenent en compte la distorsion
    vector<Pt2dr> aVp;
    Pt2dr p1 = mCam->R3toF2(P1);
    Pt2dr p2 = mCam->R3toF2(P2);
    Pt2dr p3 = mCam->R3toF2(P3);
    Pt2dr p4 = mCam->R3toF2(P4);
    aVp.push_back(p1);    aVp.push_back(p2);    aVp.push_back(p3);    aVp.push_back(p4);

    cout<<" + Correspondant : p_backproj,P_3D "<<endl;
    cout<<p1<<P1<<endl;
    cout<<p2<<P2<<endl;
    cout<<p3<<P3<<endl;
    cout<<p4<<P4<<endl;

    ElPackHomologue  aPackHomImTer;
    aPackHomImTer.Cple_Add(ElCplePtsHomologues(p1, Pt2dr(P1.x,P1.y)));
    aPackHomImTer.Cple_Add(ElCplePtsHomologues(p2, Pt2dr(P2.x,P2.y)));
    aPackHomImTer.Cple_Add(ElCplePtsHomologues(p3, Pt2dr(P3.x,P3.y)));
    aPackHomImTer.Cple_Add(ElCplePtsHomologues(p4, Pt2dr(P4.x,P4.y)));

    // define the homography
    cElHomographie H(aPackHomImTer,true);
    //H = cElHomographie::RobustInit(qual,aPackHomImTer,bool Ok(1),1, 1.0,4);
    // keep the inverse of the homography, as it this used to transform terrain coordinates to image coordinates
    cElHomographie aHInv=H.Inverse();

    cout<<" + Homography by 4 coins images : "<<endl;
    H.Show();
    cout<<endl;

    Pt2di aSz(mCam->SzPixel());
    double aGSDInit = mCam->ResolutionAngulaire() * mCam->GetProfondeur();
    cout<<" + GSD : "<<aGSDInit<<endl; // metre/pixel
    // Box sol :
    Pt2dr aBoxSolMin(ElMin4(aVP[0].x,aVP[1].x,aVP[2].x,aVP[3].x), ElMin4(aVP[0].y,aVP[1].y,aVP[2].y,aVP[3].y));
    Pt2dr aBoxSolMax(ElMax4(aVP[0].x,aVP[1].x,aVP[2].x,aVP[3].x), ElMax4(aVP[0].y,aVP[1].y,aVP[2].y,aVP[3].y));
    //cout<<"Box sol : "<<aBoxSolMin<<aBoxSolMax<<aBoxSolMax-aBoxSolMin<<endl;
    // Determiner offset georef et taille d'image rectifié :
    Pt2dr offset(aBoxSolMin.x, -aBoxSolMax.y);
    Pt2di aPImg;
    // compute homographie by 100 points
//    for (aPImg.x=0; aPImg.x<aSz.x; aPImg.x=aPImg.x+200)
//    {
//        for (aPImg.y=0; aPImg.y<aSz.y; aPImg.y=aPImg.y+200)
//        {
//            Pt3dr aPTer (aPImg.x*aGSDInit + aBoxSolMin.x,  aPImg.y*aGSDInit + aBoxSolMin.y, mZTerrain);
//            Pt2dr aPImgProj = mCam->R3toF2(aPTer);
//            aPackHomImTer.Cple_Add(ElCplePtsHomologues(aPImgProj, Pt2dr(aPTer.x,aPTer.y)));
//        }
//    }
//    H = cElHomographie(aPackHomImTer,true);
//    aHInv=H.Inverse();
//    cout<<" + Homography by more pts : "<<endl;
//    H.Show();
//    cout<<endl;

    // rectify image
    cISR_ColorImg ImCol(mNameTiff.c_str());
    cISR_ColorImg ImColRect(aSz);
    for (aPImg.x=0; aPImg.x<aSz.x; aPImg.x++)
    {
        for (aPImg.y=0; aPImg.y<aSz.y; aPImg.y++)
        {
            Pt2dr aPTerPlani(aPImg.x*aGSDInit + aBoxSolMin.x,  aPImg.y*aGSDInit + aBoxSolMin.y);
            Pt2dr aPImHomoGr = aHInv.Direct(aPTerPlani);
            if (aPImHomoGr.x > 0 && aPImHomoGr.y > 0 && aPImHomoGr.x < aSz.x && aPImHomoGr.y < aSz.y)
            {
                cISR_Color aCol=ImCol.getr(aPImHomoGr);
                // write the value on the rectified image
                //cout<<aPTerPlani<<aPImHomoGr<<endl;
                ImColRect.set(aPImg,aCol);
            }
        }
    }
    this->WriteImage(ImColRect, "HomoGr4Pts");
    // write the tfw file
    this->GenTFW(aGSDInit, offset, "Proj4Pts");
}



void cISR_Ima::InitHomography()
{
    cout<<"Init Homography..."<<endl;

    // generate 100 homol couples linking image geometry and planimetric (terrain) geometry, distributed accros the image, used for determing the homography
    ElPackHomologue  aPackHomImTer;
    Pt2di aP;
    // Loop through the terrain space, 10 times (x) x 10 times (y)

    cout<<"mBorder : "<<mBorder[0]<<" "<<mBorder[1]<<" "<<mBorder[2]<<" "<<mBorder[3]<<endl;
    cout<<"mIGSD : "<<mIGSD<<endl;
    cout<<"Pt Terrain compute.."<<endl;
    for (aP.x=0 ; aP.x<mCam->Sz().x; aP.x += (mCam->Sz().x/10))
    {
        // compute X coordinate in ground/object geometry
        double aX=mBorder[0]+mIGSD * aP.x;  // X_img*GSD = X_metric; X_metric + Origin(mBorder[0]) = X_ter

        for (aP.y=0 ; aP.y<mCam->Sz().y; aP.y += (mCam->Sz().y/10))
        {
            // compute Y coordinate in ground/object geometry
            double aY=mBorder[3]-mIGSD * aP.y;  // Y_img*GSD = Y_metric; Y_metric + Origin(mBorder[3]) = Y_ter
            // define the point position in ground geometry
            Pt3dr aPTer(aX,aY,mZTerrain);   // combine (X_ter, Y_ter , Z) = P_Ter (Z is vol altitude)
            // project this point in the initial image
            Pt2dr aPIm0 = mCam->R3toF2(aPTer);
            //std::cout << " point 2D image  :: " << aPIm0 <<"    \n";
            //std::cout << " point 3D terrain :: " << aPTer <<"    \n";

            ElCplePtsHomologues Homol(aPIm0,Pt2dr (aX,aY));
            // add the homol cple in the homol pack
            aPackHomImTer.Cple_Add(Homol);

            //cout<<"Pt2D : "<<aP<<" -- Pt3D : "<<aPTer<<endl;
        }
    }
    
    // define the homography
    cElHomographie H(aPackHomImTer,true);
    //H = cElHomographie::RobustInit(qual,aPackHomImTer,bool Ok(1),1, 1.0,4);
    // keep the inverse of the homography, as it this used to transform terrain coordinates to image coordinates
    mH=H.Inverse();
    cout<<" + Homography by 100 pts : "<<endl;
    H.Show();
    cout<<endl;
}

void cISR_Ima::ChangeGeomTerrain()
{
    // used when QuickResampling=1
    mSzImRect = mSzImRect/mDeZoom;
    mLoopGSD = mLoopGSD*mDeZoom;
}

/********************************************************************/
/*                                                                  */
/*         cISR_Appli                                               */
/*                                                                  */
/********************************************************************/


cISR_Appli::cISR_Appli(int argc, char** argv){
    // Reading parameter : check and  convert strings to low level objects
    mShowArgs=false;
    int mFlightAlti = 0;
    int mDeZoom=4;
    std::string mPrefixOut="Rectified";
    bool mByHomography=true;
    bool mQuickResampling=true;
    bool mDoMosaic = false;
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(mFullName,"Full Name (Dir+Pat)")
                << EAMC(mOri,"Used orientation, must be a projected coordinate system (no WGS, relative or RTL orientation). If DoOC=1, give text file here"),
                LArgMain()  << EAM(mPrefixOut,"PrefixOut",true,"Prefix for the name of the resulting rectified image (ex 'toto' --> toto-R000567.JPG.tif), by default == 'Rectified'")
                << EAM(mFlightAlti,"FAlti",true,"The flight altitude Above Ground Level. By default, use the flight alti computed by aerotriangulation")
                << EAM(mDeZoom,"DeZoom",true,"DeZoom of the original image, by default dezoom 4")
                << EAM(mByHomography,"ByHomography",true,"Perform the image rectification by homography? Default true, quicker but less accurate")
                << EAM(mQuickResampling,"QuickResampling",true,"Handle the resampling with a quick but non-adequate resample technique (default=true)")
                << EAM(mShowArgs,"Show",true,"Print details during the processing")
                << EAM(mDoMosaic, "Mosaic",true,"Do mosaic")
                   );
    // Initialize name manipulator & files
    SplitDirAndFile(mDir,mPat,mFullName);
    // define the "working directory" of this session
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    // create the list of images starting from the regular expression (Pattern)
    mLFile = mICNM->StdGetListOfFile(mPat);

    StdCorrecNameOrient(mOri,mDir);

    // the optional argument Show = True, print the number of images as well as the names of every images
    if (mShowArgs) DoShowArgs1();
    
    // Initialize the images list in the class cISR_Ima
    for (
         std::list<std::string>::iterator itS=mLFile.begin();
         itS!=mLFile.end();
         itS++
         )
    {
        cISR_Ima * aNewIm = new  cISR_Ima(*this,*itS,mFlightAlti,mDeZoom,mPrefixOut,mQuickResampling);
        mIms.push_back(aNewIm);

        //test if there are enough information about flight altitude (either given by the aerotriangulation or the user with FAlti)

        if (mFlightAlti==0 && (aNewIm->DepthIsDefined()==0)) ELISE_ASSERT(false,"Flight Altitude not known (use FAlti)");
    }

    long start=time(NULL);

    if (! mDoMosaic)
    {
//        for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
//        {
//            cISR_Ima * anIm = mIms[aKIm];
//            //anIm->Estime4PtsProjectiveTransformation();
//            anIm->RectifyByHomography();
//        }
        // Define the ground footprint (image swath) of every rectified images
        Appli_InitGeomTerrain();
        if (mQuickResampling) Appli_ChangeGeomTerrain();

        if (mByHomography){
            // compute the homography relation
            Appli_InitHomography();
            // Compute all rectified images
            Appli_ApplyImHomography(mShowArgs);

        } else {
            // Compute all rectified images
            Appli_ApplyImProj(mShowArgs);
        }

        long end = time(NULL);
        if (mShowArgs) cout<<"Rectification computed in "<<end-start<<" sec"<<endl;

    }
    else
    {
        cout.precision(10);
        double aMinX=1e30, aMinY=1e30, aMaxX=1e30, aMaxY=1e30; // Warn un init => put 1e30
        double aMinGSDX=1e30, aMinGSDY=1e30, aMaxGSDX=1e30, aMaxGSDY=1e30; // Warn un init => put 1e30
        for (uint aKIm = 0; aKIm<(mIms.size()) ; aKIm++)
        {
            cISR_Ima * anIm = mIms[aKIm];
            std::string aNameTFW = "Rectified-"+ anIm->Name() + ".tfw";
            std::ifstream aFp(aNameTFW);
            double a;
            int cnt = 0;

            double aGSDx = 1e30, aGSDy = 1e30, aOffsetX = 1e30, aOffsetY = 1e30; // Warn un init => put 1e30
            while (aFp >> a)
            {
                cnt++;
                if (cnt == 1)
                    aGSDx = a;
                if (cnt == 4)
                    aGSDy = a;
                if (cnt == 5)
                    aOffsetX = a;
                if (cnt == 6)
                    aOffsetY = a;
            }
            aFp.close();
            if (aKIm == 0)
            {
                aMinX = aOffsetX;
                aMaxX = aOffsetX;
                aMaxY = aOffsetY;
                aMinY = aOffsetY;
                aMinGSDX = aGSDx;
                aMaxGSDX = aGSDx;
                aMaxGSDY = aGSDy;
                aMinGSDY = aGSDy;
            }
            else
            {
                if (aOffsetX < aMinX)
                    aMinX = aOffsetX;
                if (aOffsetY < aMinY)
                    aMinY = aOffsetY;
                if (aOffsetX > aMaxX)
                    aMaxX = aOffsetX;
                if (aOffsetY > aMaxY)
                    aMaxY = aOffsetY;
                if (aGSDx < aMinGSDX)
                    aMinGSDX = aGSDx;
                if (aGSDy < aMinGSDY)
                    aMinGSDY = aGSDy;
                if (aGSDx > aMaxGSDX)
                    aMaxGSDX = aGSDx;
                if (aMaxGSDY > aGSDy)
                    aMaxGSDY = aGSDy;
            }
        }
        cout<<"Off Min "<<aMinX<<" "<<aMinY<<endl;
        cout<<"Off Max "<<aMaxX<<" "<<aMaxY<<endl;
        cout<<"GSD Min "<<aMinGSDX<<" "<<aMinGSDY<<endl;
        cout<<"GSD Max "<<aMaxGSDX<<" "<<aMaxGSDY<<endl;
    }


    // Define the ground footprint (image swath) of every rectified images
//    Appli_InitGeomTerrain();
//    if (mQuickResampling) Appli_ChangeGeomTerrain();

//    if (mByHomography){
//        // compute the homography relation
//        Appli_InitHomography();
//        // Compute all rectified images
//        Appli_ApplyImHomography(mShowArgs);

//    } else {
//        // Compute all rectified images
//        Appli_ApplyImProj(mShowArgs);
//    }

//    long end = time(NULL);
//    if (mShowArgs) cout<<"Rectification computed in "<<end-start<<" sec"<<endl;


}

void cISR_Appli::Appli_InitGeomTerrain()
{
    for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
    {
        cISR_Ima * anIm = mIms[aKIm];
        // Define the ground footprint of each georectified images
        anIm->InitGeomTerrain() ;
    }
}

void cISR_Appli::Appli_InitHomography()
{
    for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
    {
        cISR_Ima * anIm = mIms[aKIm];
        // Define the ground footprint of each georectified images and compute the Homography
        anIm->InitHomography();
    }
}

void cISR_Appli::Appli_ApplyImProj(bool aShow)
{
    for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
    {
        if (aShow) DoShowArgs2(aKIm);
        cISR_Ima * anIm = mIms[aKIm];
        anIm->ApplyImProj();
    }
}

void cISR_Appli::Appli_ApplyImHomography(bool aShow)
{
    for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
    {
        if (aShow) DoShowArgs2(aKIm);
        cISR_Ima * anIm = mIms[aKIm];
        anIm->ApplyImHomography();
    }
}

void cISR_Appli::Appli_ChangeGeomTerrain()
{
    for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
    {
        cISR_Ima * anIm = mIms[aKIm];
        anIm->ChangeGeomTerrain();
    }
}

std::string cISR_Appli::NameIm2NameOri(const std::string & aNameIm) const
{
    return mICNM->Assoc1To1
            (
                "NKS-Assoc-Im2Orient@-"+mOri+"@",
                aNameIm,
                true
                );
}

void cISR_Appli::DoShowArgs1()
{
    std::cout << "DIR=" << mDir << " Pat=" << mPat << " Orient=" << mOri<< "\n";
    std::cout << "Nb Files " << mLFile.size() << "\n";
    for (
         std::list<std::string>::iterator itS=mLFile.begin();
         itS!=mLFile.end();
         itS++
         )
    {
        std::cout << "    F=" << *itS << "\n";
    }
}

void cISR_Appli::DoShowArgs2(int aKIm)
{
    cISR_Ima * anIm = mIms[aKIm];
    std::cout << "Image : " << anIm->Name() << " --------- \n";
    std::cout << "DeZoom : " << anIm->mDeZoom << "\n";
    //std::cout << "QuickResampling :	" << mQuickResampling << "\n";
    std::cout << "Flight altitude [m]: 	" << anIm->mAlti << "  \n";
    std::cout << "Altitude of the gound surface  : 	" << anIm->mZTerrain << " \n";
    std::cout << "Initial Ground Sample Distance :	" << anIm->mIGSD << " \n";
    std::cout << "Ground Sample Distance of Rectified images : " << anIm->mFGSD << " \n";
    std::cout << "Rectified image size in pixel : " <<  anIm->SzXY() <<"  \n";
    std::cout << "Rectified image X coverage [m] : 	" <<  anIm->mBorder[1]-anIm->mBorder[0] << "  \n";
}

int ImageRectification(int argc,char ** argv)
{
    cISR_Appli anAppli(argc,argv);
    return EXIT_SUCCESS;
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
