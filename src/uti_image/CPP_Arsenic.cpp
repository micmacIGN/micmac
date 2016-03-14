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

    MicMa cis an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

#include "../../include/StdAfx.h"
#include "../src/uti_image/Arsenic.h"
#include "hassan/reechantillonnage.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <math.h>

void Arsenic_Banniere()
{
    std::cout <<  "\n";
    std::cout <<  " **********************************\n";
    std::cout <<  " *     A-utomated                 *\n";
    std::cout <<  " *     R-adiometric               *\n";
    std::cout <<  " *     S-hift                     *\n";
    std::cout <<  " *     E-qualization and          *\n";
    std::cout <<  " *     N-ormalization for         *\n";
    std::cout <<  " *     I-nter-image               *\n";
    std::cout <<  " *     C-orrection                *\n";
    std::cout <<  " **********************************\n\n";
}

string FindMaltEtape(int ResolModel, std::string aNameIm, std::string aPatModel)
{
        //Getting full image size
        Tiff_Im aTFforSize= Tiff_Im::StdConvGen(aNameIm,1,false);
        int aSzX = aTFforSize.sz().x;

        std::string aDir,aPat;
        SplitDirAndFile(aDir,aPat,aPatModel);

        cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
        const std::vector<std::string> * aSetModel = aICNM->Get(aPat);

        std::vector<std::string> aVectModel=*aSetModel;
        int nbModel = (int)aVectModel.size();

        int aEtape = 0;
        for (int i=0 ; i<nbModel ; i++)
        {
            cElNuage3DMaille * info3D = cElNuage3DMaille::FromFileIm(aDir+aVectModel[i]);
            //cout<<info3D->SzGeom()<<endl;
            int ResolThisFile=float(aSzX)/float(info3D->SzGeom().x)+0.5;
            //cout<<"ResolThisFile : "<<ResolThisFile<<endl;
            if(ResolModel==ResolThisFile){aEtape=i+1;}
        }
    cout<<"MicMac step to be used = num"<<aEtape<<endl;

    // Modif GM: compilation c++11
    ostringstream oss;
    oss << aEtape;
    return oss.str();

    //string aEtapeStr = static_cast<ostringstream*>( &(ostringstream() << aEtape) )->str();
    //return aEtapeStr;
}

vector<ArsenicImage> LoadGrpImages(string aDir, std::string aPatIm, int ResolModel, string InVig)
{
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);
    std::vector<std::string> aVectIm=*aSetIm;

    //Scaling the images the comply with MM-Initial-Model
    list<string> ListConvert, ListVig;
    vector<std::string> VectImSc,VectMasq;
    int nbIm = (int)aVectIm.size();
    char ResolModelch[3];sprintf(ResolModelch, "%02d", ResolModel);string ResolModelStr=(string)ResolModelch;
    if(ResolModel<10){ResolModelStr=ResolModelStr.substr(1,1);}
    //If a vignette correction folder is entered
    string postfix="";
    if(InVig!=""){
        string cmdVig=MMDir() + "bin/mm3d Vodka \"" + aPatIm + "\" DoCor=1 Out=" + InVig + " InCal=" + InVig;
        postfix="_Vodka.tif";
        ListVig.push_back(cmdVig);
        cEl_GPAO::DoComInParal(ListVig,aDir + "MkVig");
    }

    //Finding the appropriate NuageImProf_STD-MALT_Etape_[0-9].xml for the ResolModel :
    string aEtape=FindMaltEtape(ResolModel, aDir + (aVectIm)[0], "MM-Malt-Img-" + StdPrefix(aVectIm[0]) + "/NuageImProf_STD-MALT_Etape_[0-9].xml");

    //Reading images and Masq
    for (int aK1=0 ; aK1<nbIm ; aK1++)
    {
        string cmdConv=MMDir() + "bin/ScaleIm " + InVig + (aVectIm)[aK1] + postfix + " " + ResolModelStr + " F8B=1 Out=" + (aVectIm)[aK1] + "_Scaled.tif";
        ListConvert.push_back(cmdConv);

        //VectMasq.push_back("Masq-TieP-" + (aVectIm)[aK1] + "/RN" + (aVectIm)[aK1] + "_Masq.tif");
        VectMasq.push_back("MM-Malt-Img-" + StdPrefix((aVectIm)[aK1]) + "/AutoMask_STD-MALT_Num_" + aEtape + ".tif");
        //VectMasq.push_back("MM-Malt-Img-" + StdPrefix((aVectIm)[aK1]) + "/Masq_STD-MALT_DeZoom" + ResolModelStr + ".tif");
        //cout<<VectMasq[aK1]<<endl;
        VectImSc.push_back((aVectIm)[aK1]+std::string("_Scaled.tif"));
    }
    cEl_GPAO::DoComInParal(ListConvert,aDir + "MkScale");


    //Reading the infos
    vector<ArsenicImage> aGrIm;

    for (int aK1=0 ; aK1<int(nbIm) ; aK1++)
    {
        ArsenicImage aIm;
        //reading 3D info
        //cElNuage3DMaille * info3D1 = cElNuage3DMaille::FromFileIm("MM-Malt-Img-" + StdPrefix(aVectIm[aK1]) + "/NuageImProf_STD-MALT_Etape_1.xml");
        cElNuage3DMaille * info3D1 = cElNuage3DMaille::FromFileIm("MM-Malt-Img-" + StdPrefix(aVectIm[aK1]) + "/NuageImProf_STD-MALT_Etape_" + aEtape + ".xml");

        aIm.info3D=info3D1;

        Tiff_Im aTF1= Tiff_Im::StdConvGen(aDir + VectImSc[aK1],3,false);
        Tiff_Im aTFM= Tiff_Im::StdConvGen(aDir + VectMasq[aK1],1,false);
        Pt2di aSz = aTF1.sz();
        Im2D_REAL4  aIm1R(aSz.x,aSz.y);
        Im2D_REAL4  aIm1G(aSz.x,aSz.y);
        Im2D_REAL4  aIm1B(aSz.x,aSz.y);
        Im2D_INT1  aMasq(aSz.x,aSz.y);
        ELISE_COPY
            (
                aTF1.all_pts(),
                aTF1.in(),
                Virgule(aIm1R.out(),aIm1G.out(),aIm1B.out())
            );

        ELISE_COPY
            (
                aTFM.all_pts(),
                aTFM.in(),
                aMasq.out()
            );

        aIm.Mask=aMasq;
        aIm.RChan=aIm1R;
        aIm.GChan=aIm1G;
        aIm.BChan=aIm1B;
        aIm.SZ=aSz;
        aGrIm.push_back(aIm);
    }

    return aGrIm;
}

double Dist3d(Pt3d<double> aP1, Pt3d<double> aP2 ){
    return (double)std::sqrt(pow(double(aP1.x-aP2.x),2)+pow(double(aP1.y-aP2.y),2)+pow(double(aP1.z-aP2.z),2));
}

cl_MatPtsHom ReadPtsHom3D(string aDir,string aPatIm, string InVig, int ResolModel, double TPA)
{
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);
    std::vector<std::string> aVectIm=*aSetIm;
    int nbIm = (int)aVectIm.size();
    cl_MatPtsHom aMatPtsHomol;

    //Loading all images and usefull metadata (masks...)
    vector<ArsenicImage> aGrIm=LoadGrpImages(aDir, aPatIm, ResolModel, InVig);
    std::cout<<"===== "<<aGrIm.size()<< " images loaded"<<endl;

    //string PtsTxt="PtsTxt.txt";
    //ofstream file_out(PtsTxt, ios::out | ios::app);

    //going throug each pair of different images
    for (int aK1=0 ; aK1<nbIm ; aK1++)
    {
        //Creating the tie points vector
        cl_PtsRadio aPtsRadio;
        aMatPtsHomol.aMat.push_back(aPtsRadio);
        cout<<"Extracting point from image "<<aK1+1<<" out of "<<nbIm<<endl;
        //going throug each point of image
        for (int aY=0 ; aY<aGrIm[aK1].SZ.y  ; aY++)
        {
            for (int aX=0 ; aX<aGrIm[aK1].SZ.x  ; aX++)
            {
                Pt2dr pos2DPtIm1;pos2DPtIm1.x=aX;pos2DPtIm1.y=aY;
                if(aGrIm[aK1].Mask.data()[aY][aX]==0){continue;}else{//If pts in masq, go look for 3D position
                    Pt3d<double> pos3DPtIm1=aGrIm[aK1].info3D->PreciseCapteur2Terrain(pos2DPtIm1);
                        //Testing the position of the point in other images
                        vector<double> distances(nbIm,10000000); //distances between original 3D point and reprojection from other images (init to "a lot")
                        vector<Pt2dr> pos2DOtherIm(nbIm);
                        for (int aK2=0 ; aK2<int(nbIm) ; aK2++)
                        {
                            if (aK1!=aK2)// && aGrIm[aK2].info3D->PIsVisibleInImage(pos3DPtIm1))
                            {
                            Pt2dr pos2DPtIm2=aGrIm[aK2].info3D->Ter2Capteur(pos3DPtIm1);
                            //if pt in image and in masq, go look for 2D position, then 3D position
                            if(pos2DPtIm2.x>0 && pos2DPtIm2.x<aGrIm[aK2].SZ.x && pos2DPtIm2.y>0 && pos2DPtIm2.y<aGrIm[aK2].SZ.y){
                                        if(aGrIm[aK2].Mask.data()[int(pos2DPtIm2.y)][int(pos2DPtIm2.x)]){
                                            pos2DOtherIm[aK2]=pos2DPtIm2;
                                            Pt3d<double> pos3DPtIm2=aGrIm[aK2].info3D->PreciseCapteur2Terrain(pos2DPtIm2);
                                            //Compute Distance between the 2 3D points to check if they are the same ones (occlusion, beware!)
                                            distances[aK2]=Dist3d(pos3DPtIm1,pos3DPtIm2);
                                        }
                                    }
                            }else{aMatPtsHomol.aMat[aK1].SZ=aGrIm[aK1].SZ;}
                        }
                        for (int aK2=0 ; aK2<int(nbIm) ; aK2++){
                            if(distances[aK2]<(aGrIm[aK1].info3D->ResolSolOfPt(pos3DPtIm1))/TPA){//id pos3DPtIm1~=pos3DPtIm2 -->pt is considered homologous,it is added to PtsHom (Gr1, R1, G1, B1, X1, Y1, idem 2, NbPtsCouple++)
                                //Go looking for grey value of the point for each chan (Reechantillonnage/interpolation because pos2DPtIm not always integer)
                                //double Red1   =Reechantillonnage::biline(aGrIm[aK1].RChan.data(), aGrIm[aK1].SZ.x, aGrIm[aK1].SZ.y, pos2DPtIm1);
                                //double Green1 =Reechantillonnage::biline(aGrIm[aK1].GChan.data(), aGrIm[aK1].SZ.x, aGrIm[aK1].SZ.y, pos2DPtIm1);
                                //double Blue1  =Reechantillonnage::biline(aGrIm[aK1].BChan.data(), aGrIm[aK1].SZ.x, aGrIm[aK1].SZ.y, pos2DPtIm1);
                                double Red1   =aGrIm[aK1].RChan.data()[int(pos2DPtIm1.y)][int(pos2DPtIm1.x)];
                                double Green1 =aGrIm[aK1].GChan.data()[int(pos2DPtIm1.y)][int(pos2DPtIm1.x)];
                                double Blue1  =aGrIm[aK1].BChan.data()[int(pos2DPtIm1.y)][int(pos2DPtIm1.x)];
                                double Red2   =Reechantillonnage::biline(aGrIm[aK2].RChan.data(), aGrIm[aK2].SZ.x, aGrIm[aK2].SZ.y, pos2DOtherIm[aK2]);
                                double Green2 =Reechantillonnage::biline(aGrIm[aK2].GChan.data(), aGrIm[aK2].SZ.x, aGrIm[aK2].SZ.y, pos2DOtherIm[aK2]);
                                double Blue2  =Reechantillonnage::biline(aGrIm[aK2].BChan.data(), aGrIm[aK2].SZ.x, aGrIm[aK2].SZ.y, pos2DOtherIm[aK2]);
                                aMatPtsHomol.aMat[aK1].Pts.push_back(pos2DPtIm1.mul(ResolModel));
                                if(Red1>0){aMatPtsHomol.aMat[aK1].kR.push_back((1 + Red2/Red1 )/2);}else{aMatPtsHomol.aMat[aK1].kR.push_back((1 + Red2)/2);}
                                if(Green1>0){aMatPtsHomol.aMat[aK1].kG.push_back((1 + Green2/Green1 )/2);}else{aMatPtsHomol.aMat[aK1].kG.push_back((1 + Green2)/2);}
                                if(Blue1>0){aMatPtsHomol.aMat[aK1].kB.push_back((1 + Blue2/Blue1 )/2);}else{aMatPtsHomol.aMat[aK1].kB.push_back((1 + Blue2)/2);}
                                aMatPtsHomol.aMat[aK1].OtherIm.push_back(aK2);
                                aMatPtsHomol.aMat[aK1].SZ=aGrIm[aK1].SZ;
                                //file_out <<(Red2+Blue2+Green2)/(Red1+Blue1+Green1)<<endl;
                            }
                        }
                }
            }
        }
    }
    //file_out.close();
    return aMatPtsHomol;

}

cl_MatPtsHom TiePtsFilter(cl_MatPtsHom aMatPtsHomol, double aThresh)
{
    //Computing the mean of the correction factor
    int nbIm = (int)aMatPtsHomol.aMat.size();
    for (int numIm=0 ; numIm<nbIm ; numIm++)
    {
        cout<<"[Im "<< numIm <<"] NbPts before filter : "<<aMatPtsHomol.aMat[numIm].size()<<endl;
        double meanR=0,meanG=0,meanB=0;
        for(int i=0 ; i<aMatPtsHomol.aMat[numIm].size() ; i++)
        {
            meanR += aMatPtsHomol.aMat[numIm].kR[i];
            meanG += aMatPtsHomol.aMat[numIm].kG[i];
            meanB += aMatPtsHomol.aMat[numIm].kB[i];
        }
        meanR = meanR/aMatPtsHomol.aMat[numIm].size(); cout<<"Mean R = "<<meanR<<endl;
        meanG = meanG/aMatPtsHomol.aMat[numIm].size(); cout<<"Mean G = "<<meanG<<endl;
        meanB = meanB/aMatPtsHomol.aMat[numIm].size(); cout<<"Mean B = "<<meanB<<endl;

        //If a factor is different by more than "seuil" from the mean, the point is considered an outlier
        for(int i=aMatPtsHomol.aMat[numIm].size()-1 ; i>=0 ; i--)
        {
            if(aMatPtsHomol.aMat[numIm].kR[i]>meanR*aThresh || aMatPtsHomol.aMat[numIm].kR[i]<meanR/aThresh ||
               aMatPtsHomol.aMat[numIm].kG[i]>meanG*aThresh || aMatPtsHomol.aMat[numIm].kG[i]<meanG/aThresh ||
               aMatPtsHomol.aMat[numIm].kB[i]>meanB*aThresh || aMatPtsHomol.aMat[numIm].kB[i]<meanB/aThresh)
            {
                aMatPtsHomol.aMat[numIm].kR.erase(aMatPtsHomol.aMat[numIm].kR.begin() + i);
                aMatPtsHomol.aMat[numIm].kG.erase(aMatPtsHomol.aMat[numIm].kG.begin() + i);
                aMatPtsHomol.aMat[numIm].kB.erase(aMatPtsHomol.aMat[numIm].kB.begin() + i);
                aMatPtsHomol.aMat[numIm].Pts.erase(aMatPtsHomol.aMat[numIm].Pts.begin() + i);
                aMatPtsHomol.aMat[numIm].OtherIm.erase(aMatPtsHomol.aMat[numIm].OtherIm.begin() + i);
            }
        }
        cout<<"[Im "<< numIm <<"] NbPts after filter : "<<aMatPtsHomol.aMat[numIm].size()<<endl;
    }
    return aMatPtsHomol;
}

void Egal_field_correct_ite(string aDir,std::vector<std::string> * aSetIm, cl_MatPtsHom aMatPtsHomol , string aDirOut, string InVig, int ResolModel, int nbIm, int nbIte, double aThresh)
{
//truc à iterer--------------------------------------------------------------------------------------------------------------------------------------
for(int iter=0;iter<nbIte;iter++){
    cout<<"Pass "<<iter+1<<" out of "<< nbIte<<endl;

    //Filtering the tie points
    aMatPtsHomol = TiePtsFilter(aMatPtsHomol, aThresh);

//Correcting the tie points

//#pragma omp parallel for

    for(int numImage1=0;numImage1<nbIm;numImage1++)
    {
        vector<int> cpt(nbIm,0);
        cout<<"Computing factors for Im "<<numImage1<<endl;

        //For each tie point point, compute correction value (distance-ponderated mean value of all the tie points)
        for(int k = 0; k<int(aMatPtsHomol.aMat[numImage1].size()) ; k++){//go through each tie point
            double aCorR=0.0,aCorG=0.0,aCorB=0.0;
            double aSumDist=0;
            Pt2dr aPt(aMatPtsHomol.aMat[numImage1].Pts[k].x/ResolModel,aMatPtsHomol.aMat[numImage1].Pts[k].x/ResolModel);
            for(int numPt = 0; numPt<int(aMatPtsHomol.aMat[numImage1].size()) ; numPt++){//go through each tie point
                Pt2dr aPtIn(aMatPtsHomol.aMat[numImage1].Pts[numPt].x/ResolModel,aMatPtsHomol.aMat[numImage1].Pts[numPt].y/ResolModel);
                double aDist=euclid(aPtIn, aPt);
                if(aDist<1){aDist=1;}
                aSumDist=aSumDist+1/(aDist);
                aCorR = aCorR + aMatPtsHomol.aMat[numImage1].kR[numPt]/(aDist);
                aCorG = aCorG + aMatPtsHomol.aMat[numImage1].kG[numPt]/(aDist);
                aCorB = aCorB + aMatPtsHomol.aMat[numImage1].kB[numPt]/(aDist);
            }
            //Normalize
            aCorR = aCorR/aSumDist;
            aCorG = aCorG/aSumDist;
            aCorB = aCorB/aSumDist;

            //correcting Tie points color with computed surface
            //int numImage2=aMatPtsHomol.aMat[numImage1].OtherIm[k];
            //int pos=cpt[numImage2];cpt[numImage2]++;
            //if(aMatPtsHomol.aMat[numImage1][numImage2].R1[pos]*aCorR>255)
            //{
            //	aCorR=255/aMatPtsHomol.aMat[numImage1][numImage2].R1[pos];
            //}
            //if(aMatPtsHomol.aMat[numImage1][numImage2].G1[pos]*aCorB>255)
            //{
            //	aCorG=255/aMatPtsHomol.aMat[numImage1][numImage2].G1[pos];
            //}
            //if(aMatPtsHomol.aMat[numImage1][numImage2].B1[pos]*aCorG>255)
            //{
            //	aCorB=255/aMatPtsHomol.aMat[numImage1][numImage2].B1[pos];
            //}
            aMatPtsHomol.aMat[numImage1].kR[k]=aCorR;
            aMatPtsHomol.aMat[numImage1].kG[k]=aCorG;
            aMatPtsHomol.aMat[numImage1].kB[k]=aCorB;
            }
        //cout<<cpt<<endl;

    }
}

//Filtering the tie points
aMatPtsHomol = TiePtsFilter(aMatPtsHomol, aThresh);

cout<<"Factors were computed"<<endl;
//end truc à iterer--------------------------------------------------------------------------------------------------------------------------------------



//Applying the correction to the images
    //Bulding the output file system
    ELISE_fp::MkDirRec(aDir + aDirOut);
    //Reading input files
    string suffix="";if(InVig!=""){suffix="_Vodka.tif";}


#ifdef USE_OPEN_MP
#pragma omp parallel for
#endif
    for(int i=0;i<nbIm;i++)
    {
        string aNameIm=InVig + (*aSetIm)[i] + suffix;//if vignette is used, change the name of input file to read
        cout<<"Correcting "<<aNameIm<<" (with "<<aMatPtsHomol.aMat[i].size()<<" data points)"<<endl;
        string aNameOut=aDir + aDirOut + (*aSetIm)[i] +"_egal.tif";

        Pt2di aSzMod=aMatPtsHomol.aMat[i].SZ;//Size of the correction surface, taken from the size of the scaled image
        //cout<<"aSzMod"<<aSzMod<<endl;
        Im2D_REAL4  aImCorR(aSzMod.x,aSzMod.y,0.0);
        Im2D_REAL4  aImCorG(aSzMod.x,aSzMod.y,0.0);
        Im2D_REAL4  aImCorB(aSzMod.x,aSzMod.y,0.0);
        REAL4 ** aCorR = aImCorR.data();
        REAL4 ** aCorG = aImCorG.data();
        REAL4 ** aCorB = aImCorB.data();
        //cout<<vectPtsRadioTie[i].size()<<endl;
        //For each point of the surface, compute correction value (distance-ponderated mean value of all the tie points)
        long start=time(NULL);
        for (int aY=0 ; aY<aSzMod.y  ; aY++)
            {
                for (int aX=0 ; aX<aSzMod.x  ; aX++)
                {
                    float aCorPtR=0,aCorPtG=0,aCorPtB=0;
                    double aSumDist=0;
                    Pt2dr aPt(aX,aY);
                    for(int j = 0; j<int(aMatPtsHomol.aMat[i].size()) ; j++){//go through each tie point
                        Pt2dr aPtIn(aMatPtsHomol.aMat[i].Pts[j].x/ResolModel,aMatPtsHomol.aMat[i].Pts[j].y/ResolModel);
                        double aDist=euclid(aPtIn, aPt);
                        if(aDist<1){aDist=1;}
                        aSumDist=aSumDist+1/(aDist);
                        aCorPtR = aCorPtR + aMatPtsHomol.aMat[i].kR[j]/(aDist);
                        aCorPtG = aCorPtG + aMatPtsHomol.aMat[i].kG[j]/(aDist);
                        aCorPtB = aCorPtB + aMatPtsHomol.aMat[i].kB[j]/(aDist);
                    }
                    //Normalize
                    aCorR[aY][aX] = aCorPtR/aSumDist;
                    aCorG[aY][aX] = aCorPtG/aSumDist;
                    aCorB[aY][aX] = aCorPtB/aSumDist;
                }
            }

        long end = time(NULL);
        cout<<"Correction field computed in "<<end-start<<" sec, applying..."<<endl;

        //Reading the image and creating the objects to be manipulated
        Tiff_Im aTF= Tiff_Im::StdConvGen(aDir + aNameIm,3,false);
        Pt2di aSz = aTF.sz();

        Im2D_U_INT1  aImR(aSz.x,aSz.y);
        Im2D_U_INT1  aImG(aSz.x,aSz.y);
        Im2D_U_INT1  aImB(aSz.x,aSz.y);

        ELISE_COPY
        (
           aTF.all_pts(),
           aTF.in(),
           Virgule(aImR.out(),aImG.out(),aImB.out())
        );

        U_INT1 ** aDataR = aImR.data();
        U_INT1 ** aDataG = aImG.data();
        U_INT1 ** aDataB = aImB.data();

        for (int aY=0 ; aY<aSz.y  ; aY++)
            {
                for (int aX=0 ; aX<aSz.x  ; aX++)
                {
                    Pt2dr aPt(double(aX/ResolModel),double(aY/ResolModel));
                    //To be able to correct the edges
                        if(aPt.x>aSzMod.x-2){aPt.x=aSzMod.x-2;}
                        if(aPt.y>aSzMod.y-2){aPt.y=aSzMod.y-2;}
                    //Bilinear interpolation from the scaled surface to the full scale image
                    double R = aDataR[aY][aX]*Reechantillonnage::biline(aCorR, aSzMod.x, aSzMod.y, aPt);
                    double G = aDataG[aY][aX]*Reechantillonnage::biline(aCorG, aSzMod.x, aSzMod.y, aPt);
                    double B = aDataB[aY][aX]*Reechantillonnage::biline(aCorB, aSzMod.x, aSzMod.y, aPt);
                    //Overrun management:
                    if(R>255){aDataR[aY][aX]=255;}else if(R<0){aDataR[aY][aX]=0;}else{aDataR[aY][aX]=R;}
                    if(G>255){aDataG[aY][aX]=255;}else if(G<0){aDataG[aY][aX]=0;}else{aDataG[aY][aX]=G;}
                    if(B>255){aDataB[aY][aX]=255;}else if(B<0){aDataB[aY][aX]=0;}else{aDataB[aY][aX]=B;}
                }
        }


        //Writing ouput image
         Tiff_Im  aTOut
            (
                aNameOut.c_str(),
                aSz,
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
}

int  Arsenic_main(int argc,char ** argv)
{

    std::string aFullPattern,aDirOut="Egal/",InVig="";
    //bool InTxt=false;
    int ResolModel=16;
    double TPA=16,aThresh=1.4;
    int nbIte=5;
      //Reading the arguments
        ElInitArgMain
        (
            argc,argv,
            LArgMain()  << EAMC(aFullPattern,"Images Pattern", eSAM_IsPatFile),
            LArgMain()  << EAM(aDirOut,"Out",true,"Output folder (end with /) and/or prefix (end with another char)")
                        << EAM(InVig,"InVig",true,"Input vignette folder (for example : Vignette/ )", eSAM_IsDir)
                        << EAM(ResolModel,"ResolModel",true,"Resol of input model (Def=16)")
                        << EAM(TPA,"TPA",true,"Tie Point Accuracy (Higher is better, lower gives more points Def=16)")
                        << EAM(nbIte,"NbIte",true,"Number of iteration of the process (default=5)")
                        << EAM(aThresh,"ThreshDisp",true,"Disparity threshold between the tie points (Def=1.4 for 40%)")
        );

        if (!MMVisualMode)
        {
            std::string aDir,aPatIm;
            SplitDirAndFile(aDir,aPatIm,aFullPattern);

            cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
            const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);

            std::vector<std::string> aVectIm=*aSetIm;
            int nbIm = (int)aVectIm.size();

            ELISE_ASSERT(nbIm>1,"Less than two images found with this pattern");

            //Computing homologous points
            cout<<"Computing homologous points"<<endl;
            cl_MatPtsHom aMatPtsHomol=ReadPtsHom3D(aDir, aPatIm, InVig, ResolModel, TPA);

            //Computing and applying the equalization surface
            cout<<"Computing and applying the equalization surface"<<endl;
            Egal_field_correct_ite(aDir, & aVectIm, aMatPtsHomol, aDirOut, InVig, ResolModel, nbIm, nbIte, aThresh);

            Arsenic_Banniere();
        }

        return 0;
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

