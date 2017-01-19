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
#include "StdAfx.h"
#include "TexturePacker/TexturePacker.h"
#include "GraphCut/QPBO-v1.4/QPBO.h"

void LoadTrScaleRotate
     (
          Tiff_Im &aTifIn,
          Tiff_Im &aTifOut,
          const Pt2di & aP1Int,
          const Pt2di & aP2Int,
          const Pt2di & aP1Out,
          double      aScale,  // Par ex 2 pour image 2 fois + petite
          int         aRot
     )
{
    // Tiff_Im aTifIn(aNameIn.c_str());
    // Tiff_Im aTifOut(aNameOut.c_str());

     int aNbCh = aTifIn.nb_chan();
     ELISE_ASSERT(aTifOut.nb_chan()==aNbCh,"LoadTrScaleRotate nb channel diff");


     Pt2dr aVIn = Pt2dr(aP2Int-aP1Int);
     Pt2di aSzOutInit = round_ni(aVIn / aScale);

     vector<Im2DGen *>   aVOutInit = aTifOut.VecOfIm(aSzOutInit);

     ELISE_COPY
     (
          aVOutInit[0]->all_pts(),
          StdFoncChScale(aTifIn.in_proj(),Pt2dr(aP1Int),Pt2dr(aScale,aScale)),
          StdOut(aVOutInit)
     );

     vector<Im2DGen *> aVOutRotate;
     for (int aK=0 ; aK<int(aVOutInit.size()) ; aK++)
          aVOutRotate.push_back(aVOutInit[aK]->ImRotate(aRot));

     Pt2di aSzOutRotat = aVOutRotate[0]->sz();

     ELISE_COPY
     (
         rectangle(aP1Out,aP1Out+aSzOutRotat),
         trans(StdInput(aVOutRotate), -aP1Out),
         aTifOut.out()
     );
}

//update index in regions list, when a triangle is removed from mesh
//TODO : faux : on ne d�cale plus d'un lorsqu'on supprime un triangle
/*void updateIndex(int triIdx, vector < cTextureBox2d > &regions)
{
    for (unsigned int cK=0; cK < regions.size();++cK)
    {
        vector <int> vtri = regions[cK].triangles;
        for (unsigned int dK=0; dK < vtri.size();++dK)
        {
            if (vtri[dK] > triIdx) regions[cK].triangles[dK]--;
        }
    }
}*/

typedef enum
{
  eBasic,
  ePack,
  eLastTM
} eTequilaMode;

typedef enum
{
    eAngle,
    eStretch,
    eAAngle, //angle aigu
    eLastTC
} eTequilaCrit;

string eToString(const eTequilaMode & aVal)
{
   if (aVal==eBasic)
      return  "eBasic";
   if (aVal==ePack)
      return  "ePack";
 cout << "Enum = eTequilaMode\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

string eToString(const eTequilaCrit & aVal)
{
   if (aVal==eAngle)
      return  "eAngle";
   if (aVal==eStretch)
      return  "eStretch";
   if (aVal==eAAngle)
      return  "eAAngle";
 cout << "Enum = eTequilaCrit\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

int Tequila_main(int argc,char ** argv)
{
    string aDir, aPat, aFullName, aOri, aPly, aOut, aNameOut;
    int aTextMaxSize = 8192;
    int aZBuffSSEch = 2;
    int aJPGcomp = 70;
    double aAngleMax = 90.f;
    bool aBin = true;
    string aMode = "Pack";
    string aCrit = "Angle";
    bool aFilter = false;
    bool aDoGraphCut = false;
    double aLambda = 0.01;
    int aNbIter = 2;

    bool debug = false;
    float defValZBuf = 1e9;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullName,"Full Name (Dir+Pat)",eSAM_IsPatFile)
                            << EAMC(aOri,"Orientation path",eSAM_IsExistDirOri)
                            << EAMC(aPly,"Ply file", eSAM_IsExistFile),
                LArgMain()  << EAM(aOut,"Out",true,"Textured mesh name (def=plyName+ _textured.ply)")
                            << EAM(aBin,"Bin",true,"Write binary ply (def=true)")
                            << EAM(aDoGraphCut,"Optim",true,"Graph-cut optimization (def=false)")
                            << EAM(aLambda,"Lambda", true,"Lambda (def=0.01)")
                            << EAM(aNbIter,"Iter", true,"Optimization iteration number (def=2)")
                            << EAM(aFilter,"Filter",true,"Remove border faces (def=false)")
                            << EAM(aNameOut,"Texture",true,"Texture name (def=plyName + _UVtexture.jpg)")
                            << EAM(aTextMaxSize,"Sz",true,"Texture max size (def=4096)")
                            << EAM(aZBuffSSEch,"Scale", true, "Z-buffer downscale factor (def=2)",eSAM_InternalUse)
                            << EAM(aJPGcomp, "QUAL", true, "jpeg compression quality (def=70)")
                            << EAM(aAngleMax, "Angle", true, "Threshold angle, in degree, between triangle normal and image viewing direction (def=90)")
                            << EAM(aMode,"Mode", true, "Mode (def = Pack)", eSAM_None, ListOfVal(eLastTM))
                            << EAM(aCrit,"Crit", true, "Texture choosing criterion (def = Angle)", eSAM_None, ListOfVal(eLastTC))
             );

    if (MMVisualMode) return EXIT_SUCCESS;

    cout<<endl;
    cout<<"****************************Parameters******************************"<<endl;
    cout<<endl;

    cout << "Mode = " << aMode << endl;
    cout << "Crit = " << aCrit << endl;
    cout << "Max angle = " << aAngleMax << endl;
    cout << "Downscale factor = " << aZBuffSSEch  << endl;
    cout << "Texture max size = " << aTextMaxSize << endl;
    cout << "Write binary file = " << aBin << endl;
    cout << "jpeg compression quality = " << aJPGcomp << endl;

    cout<<endl;
    cout<<"********************************************************************"<<endl;

    SplitDirAndFile(aDir,aPat,aFullName);

    if (!EAMIsInit(&aOut)) aOut = StdPrefix(aPly) + "_textured.ply";
    if (!EAMIsInit(&aNameOut)) aNameOut = StdPrefix(aPly) + "_UVtexture.tif";

    string textureName = StdPrefix(aNameOut) + ".jpg ";
    stringstream st;
    st << aJPGcomp;

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    list<string>  aLS = aICNM->StdGetListOfFile(aPat);

    StdCorrecNameOrient(aOri,aDir);

    vector<CamStenope*> ListCam;

    cout << endl;
    for (list<string>::const_iterator itS=aLS.begin(); itS!=aLS.end() ; itS++)
    {
        string NOri=aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aOri,*itS,true);

        ListCam.push_back(CamOrientGenFromFile(NOri,aICNM));

        ListCam.back()->SetIdentCam(NOri); //Debug only

        cout <<"Image "<<*itS<<", with ori : "<< NOri <<endl;
    }

    cout<<endl;
    cout<<"**************************Reading ply file***************************"<<endl;
    cout<<endl;

    cMesh myMesh(aPly, aMode=="Pack");

    float threshold = 0.f;
    float angle_thresh = 0.f;
    float angleMax = -cos(PI*aAngleMax/180.f);//angle min = cos(180 - 60) = -0.5
    if (aCrit == "Angle") threshold = angleMax;
    else if (aCrit == "Stretch")
    {
        threshold = 1e30;
        angle_thresh = threshold;
    }
    else if (aCrit == "AAngle")
    {
        threshold = 0.f;
        angle_thresh = threshold;
    }
    //cout << "threshold=" << threshold << endl;

    myMesh.initDefValue(threshold);

    const int nFaces = myMesh.getFacesNumber();
    printf("Vertex : %d - faces : %d - edges : %d\n\n", myMesh.getVertexNumber(), nFaces, myMesh.getEdgesNumber());

    cout<<"*************************Computing Z-Buffer**************************"<< endl;
    cout<< endl;

    vector <cZBuf> aZBuffers;

    list<string>::const_iterator itS=aLS.begin();
    const int nCam = (int)ListCam.size();
    for(int aK=0 ; aK<nCam; aK++, itS++)
    {
        CamStenope* Cam = ListCam[aK];
        cout << "Z-buffer " << aK+1 << "/" << ListCam.size() << endl;

        cZBuf aZBuffer(Cam->Sz(), defValZBuf, aZBuffSSEch);

        aZBuffer.BasculerUnMaillage(myMesh, *Cam);

        aZBuffers.push_back(aZBuffer);

        set <int> vTri;
        aZBuffer.getVisibleTrianglesIndexes(vTri);

        if (debug)
        {
            aZBuffer.write(StdPrefix(*itS) + "_zbuf.tif");

            aZBuffer.writeImLabel(StdPrefix(*itS) + "_label.tif");

            myMesh.Export(StdPrefix(*itS) + "export.ply", vTri);
        }

        set <int>::const_iterator it = vTri.begin();
        for (;it!=vTri.end();++it)
        {
            cTriangle * Triangle = myMesh.getTriangle(*it);

            vector <Pt3dr> Vertex;
            Triangle->getVertexes(Vertex);

            Pt2dr D = Cam->R3toF2(Vertex[0]);             //projection des sommets du triangle
            Pt2dr E = Cam->R3toF2(Vertex[1]);
            Pt2dr F = Cam->R3toF2(Vertex[2]);

            if (Cam->IsInZoneUtile(D) && Cam->IsInZoneUtile(E) && Cam->IsInZoneUtile(F))
            {
                double criter = threshold;
                double angle  = scal(Triangle->getNormale(true), Cam->DirK()); //Norme de DirK=1

                if (aCrit == "Stretch")
                {
                    Pt2dr v1(Vertex[1].x - Vertex[0].x, Vertex[1].y - Vertex[0].y);
                    Pt2dr v2(Vertex[2].x - Vertex[0].x, Vertex[2].y - Vertex[0].y);

                    double norme = ElMax(euclid(v1), euclid(v2));

                    Pt2dr v1n = v1 / norme;
                    Pt2dr v2n = v2 / norme;

                    //coordonnees du point C dans le plan du triangle (repere orthonorm� defini par Ox = AB )
                    double x = scal(v1n, v2n);
                    double y = sin(acos(x));

                    //AB = B(1,0) - A(0,0)
                    //AC = C(x,y) - A(0,0)

                    Pt2dr DE (E - D);
                    Pt2dr DF (F - D);

                    double det = DE.x * DF.y - DF.x*DE.y;

                    double ni00 =  DF.y / det;
                    double ni10 = -DF.x / det;
                    double ni01 = -DE.y / det;
                    double ni11 =  DE.x / det;

                    //Coefficients de l'affinite
                    Pt2dr aU (ni00 , x*ni00 + y*ni10);
                    Pt2dr aV (ni01 , x*ni01 + y*ni11);

                    double aU2 = square_euclid(aU);
                    double aV2 = square_euclid(aV);
                    double aUV = scal(aU,aV);

                    // le coef d'etirement est sqrt((aU2+aV2+sqrt(ElSquare(aU2-aV2)+4*ElSquare(aUV)))/2)
                    // mais on supprime les calculs superflus => sqrt et /2 strictement monotones

                    criter = aU2+aV2+sqrt(ElSquare(aU2-aV2)+4.f*ElSquare(aUV));

                    //cout << "criter = " << criter << endl;
                    if((criter < Triangle->getBestCriter()) && (angle < angleMax))
                    {
                        Triangle->setBestImgIndex(aK);
                    }
                }
                else if (aCrit == "Angle")
                {
                    criter = angle;

                    //if (Triangle->Idx() == 14591)  cout << "aK = " << aK << " " << criter << " < " << Triangle->getBestCriter() <<  endl;

                    if(criter < Triangle->getBestCriter())
                    {
                        Triangle->setBestImgIndex(aK);
                    }
                }
                else if (aCrit == "AAngle") //max de l'angle aigu
                {
                    Pt2dr DE (E - D);
                    Pt2dr DF (F - D);
                    Pt2dr EF (F - E);

                    float DEn = euclid(DE);
                    float DFn = euclid(DF);
                    float EFn = euclid(EF);

                    float EDF = acos(scal(DE, DF) / (DEn*DFn));
                    float DFE = acos(scal(DF, EF) / (DFn*EFn));
                    float DEF = acos(scal(DE,-EF) / (DEn*EFn));

                    criter = min(EDF, min(DFE, DEF));

                      //cout << "criter = " << criter << endl;
                    if((criter > Triangle->getBestCriter()) && (angle < angle_thresh))
                    {
                        Triangle->setBestImgIndex(aK);
                    }
                }

                Triangle->insertCriter(aK,criter);
            }
        }
    }

    set <int> index; //liste des index de cameras utilisees

    int valDef = cTriangle::getDefTextureImgIndex();

    //int cotN =0;
    for (int aK=0;aK<nFaces; aK++)
    {
        int imgIdx = myMesh.getTriangle(aK)->getBestImgIndex();
        if(imgIdx != valDef) index.insert(imgIdx);
      //  else cotN++;
    }

    //cout << "faces non texturees = " << cotN << endl;

    cout << endl;
    cout << "Selected images / total : " << index.size() << " / " << aLS.size() << endl;

    if (aFilter)
    {
        cout << endl;
        cout <<"**********************Filtering border faces*************************"<<endl;
        cout << endl;

        myMesh.clean();

        printf("\nVertex number : %d - faces number : %d \n", myMesh.getVertexNumber(), myMesh.getFacesNumber());
    }

    cout << endl;
    cout <<"**************************Reading images*****************************"<<endl;
    cout << endl;

    Pt2di aSzMax;
    vector <Tiff_Im> aVT;     //Vecteur contenant les images
    int aNbCh = 0;

    vector <Im2D_REAL4> final_ZBufIm;
    cInterpolateurIm2D<REAL4> * pInterp = new cInterpolBilineaire<REAL4>;
    set <int>::const_iterator it = index.begin();
    for (; it != index.end();it++)
    {
        int bK=0;
        for (list<string>::const_iterator itS=aLS.begin(); itS!=aLS.end() ; itS++, bK++)
        {
            if (*it == bK)
            {
                aVT.push_back(Tiff_Im::StdConvGen(aDir+*itS,-1,false,true));

                if (aZBuffSSEch > 1)
                {
                    Pt2di sz = aVT.back().sz();
                    Im2D_REAL4 * pIm = new Im2D_REAL4(sz.x,sz.y,defValZBuf);
                    Im2D_REAL4 * pZBuf = aZBuffers[*it].get();
                    float **pImData = pIm->data();

                    for (int cK=0; cK < sz.x; cK++)
                        for(int dK=0; dK < sz.y; dK++)
                            pImData[dK][cK] = pZBuf->Get(Pt2dr(cK, dK) / aZBuffSSEch, *pInterp, defValZBuf);

                    final_ZBufIm.push_back(*pIm);
                }
                else
                    final_ZBufIm.push_back(*(aZBuffers[*it].get()));

                aSzMax.SetSup(aVT.back().sz());
                aNbCh = ElMax(aNbCh,aVT.back().nb_chan());
                break;
            }
        }
    }

    Tiff_Im::PH_INTER_TYPE aPhI = Tiff_Im::RGBPalette;

    if (aVT.size()) aPhI = aVT[0].phot_interp();

    /*cTriangle* trin = myMesh.getTriangle(14591);
    vector<int> som;
    trin->getVertexesIndexes(som);
    cout << "sommets = " << som[0] << " "<< som[1] << " " << som[2] << endl;
    cout << "img idx before = " << trin->getBestImgIndex() << endl;*/

    if (aDoGraphCut)
    {
        cout << endl;
        cout <<"***************************Optimization******************************"<<endl;
        cout << endl;

        Pt3dr mte(1e9,1e9,1e9); //mean texture exception (triangle projects outside image)
        float curCrit, newCrit, diff00, diff01, diff10, diff11;
        Pt3dr curMean0, curMean1, newMean0, newMean1;
        int n0, n1, curImgIdx0, curImgIdx1, newImgIdx0, newImgIdx1;

        const int nTriangles = myMesh.getFacesNumber();
        const int nEdges = myMesh.getEdgesNumber();
        if (nTriangles && nEdges)
        {
            //cout << "nb faces, nb edges = " << nTriangles << " " <<  nEdges << endl;

            for (int optIter=0; optIter < aNbIter; optIter++)
            {
                cout << "Iteration " << optIter+1 << "/" << aNbIter << endl;

                for(int aCam=0; aCam< nCam;++aCam)
                {
                    QPBO<float>* q = new QPBO<float>(nTriangles, nEdges); // max number of nodes & edges

                    set <int> vTri;
                    aZBuffers[aCam].getVisibleTrianglesIndexes(vTri);

                    //if (aCam==7 && (vTri.find(14591) != vTri.end()) ) cout << "the triangle is visible" << endl;

                    for(int aK=0; aK < nTriangles; ++aK)
                    {
                        q->AddNode(1);

                        cTriangle *tri = myMesh.getTriangle(aK);
                        int curImgIdx = tri->getBestImgIndex();

                        if ((curImgIdx != valDef) && (tri->getCriter(aCam) != threshold))
                        {
                            if (aCrit == "Angle")
                            {
                                curCrit = 1.f / tri->getCriter(curImgIdx);
                                newCrit = 1.f / tri->getCriter(aCam);
                            }
                            else
                            {
                                curCrit = tri->getCriter(curImgIdx);
                                newCrit = tri->getCriter(aCam);
                            }

                            q->AddUnaryTerm(aK, curCrit*curCrit, newCrit*newCrit);
                        }
                        else q->AddUnaryTerm(aK, 300, 300);
                    }

                    for (int aK=0; aK < nEdges; ++aK)
                    {
                        cEdge *edge = myMesh.getEdge(aK);

                        n0 = edge->n1();
                        n1 = edge->n2();

                        cTriangle *tri0 = myMesh.getTriangle(n0);
                        cTriangle *tri1 = myMesh.getTriangle(n1);

                        curImgIdx0 = tri0->getBestImgIndex();
                        curImgIdx1 = tri1->getBestImgIndex();

                        if ((curImgIdx0 != valDef) && (curImgIdx1 != valDef) &&
                            (vTri.find(n0) != vTri.end()) &&
                            (vTri.find(n1) != vTri.end()) )
                        {
                            newImgIdx0 = aCam; //tri0->getBestImgIndexAfter(curImgIdx0);
                            newImgIdx1 = aCam; //tri1->getBestImgIndexAfter(curImgIdx1);

                            curMean0 = tri0->meanTexture(ListCam[curImgIdx0], aVT[curImgIdx0]); //TODO: calculer une seule fois...
                            curMean1 = tri1->meanTexture(ListCam[curImgIdx1], aVT[curImgIdx1]);

                            newMean0 = tri0->meanTexture(ListCam[newImgIdx0], aVT[newImgIdx0]);
                            newMean1 = tri1->meanTexture(ListCam[newImgIdx1], aVT[newImgIdx1]);

                            if ((curMean0 == mte) || (curMean1 == mte) || (newMean0 == mte) || (newMean1 == mte))
                                q->AddPairwiseTerm(tri0->Idx(), tri1->Idx(), 300,300,300,300);
                            else
                            {
                                diff00 = aLambda*(fabs(curMean0.x - curMean1.x) + fabs(curMean0.y - curMean1.y) + fabs(curMean0.z - curMean1.z));
                                diff01 = aLambda*(fabs(curMean0.x - newMean1.x) + fabs(curMean0.y - newMean1.y) + fabs(curMean0.z - newMean1.z));
                                diff10 = aLambda*(fabs(newMean0.x - curMean1.x) + fabs(newMean0.y - curMean1.y) + fabs(newMean0.z - curMean1.z));
                                diff11 = aLambda*(fabs(newMean0.x - newMean1.x) + fabs(newMean0.y - newMean1.y) + fabs(newMean0.z - newMean1.z));

                                #ifdef _DEBUG
                                    cout << "image " << aCam << endl;
                                    cout << "curMean0 = " << curMean0 << " newMean0 " << newMean0 << endl;
                                    cout << "curMean1 = " << curMean1 << " newMean1 " << newMean1 << endl;

                                    cout << "diff00 = " << diff00 << endl;
                                    cout << "diff01 = " << diff01 << endl;
                                    cout << "diff10 = " << diff10 << endl;
                                    cout << "diff11 = " << diff11 << endl;

                                    cout << "tri0->getCriter(curImgIdx0)) " << tri0->getCriter(curImgIdx0) << endl;
                                    cout << "tri1->getCriter(curImgIdx0)) " << tri1->getCriter(curImgIdx1) << endl;
                                #endif

                                q->AddPairwiseTerm(tri0->Idx(), tri1->Idx(), diff00, diff01, diff10, diff11);
                            }
                        }
                        else q->AddPairwiseTerm(tri0->Idx(), tri1->Idx(), 300,300,300,300);
                    }

                    cout << "Solving for image "<< aCam+1 << "/"<< nCam << endl;
                    q->Solve();
                    q->ComputeWeakPersistencies();

                    for (int aK=0; aK < nTriangles; ++aK)
                    {
                        cTriangle *tri = myMesh.getTriangle(aK);

                        if (tri->getBestImgIndex() != valDef)
                        {
                            int x = q->GetLabel(tri->Idx());

                            //if (debug) printf("Solution: x=%d\n", x);

                            if (x==1) tri->setBestImgIndex(aCam);
                        }
                    }
                    delete q;
                }
            }
        }
        else cout << "Walou faces or walou edges" << endl;
    }


    //cout << "img idx = " << trin->getBestImgIndex() << endl;
    //cout << "***************************************************************" << endl;

    if (aMode == "Pack")
    {
        cout << endl;
        cout <<"**********************Getting adjacent triangles*********************"<<endl;
        cout << endl;

        vector < cTextureBox2d > regions = myMesh.getRegions();
        cout << regions.size() << " regions found" << endl;

        TEXTURE_PACKER::TexturePacker *tp = TEXTURE_PACKER::createTexturePacker();

        for (unsigned int aK=0; aK < regions.size();++aK)
        {
            cTextureBox2d *region = &(regions[aK]);
            //cout << "region " << aK << " nb triangles = " << regions[aK].triangles.size() << endl;
            //Calcul de la zone correspondante dans l'image

            int triIdx = region->triangles[0];
            cTriangle * Tri = myMesh.getTriangle(triIdx);
            int imgIdx = Tri->getBestImgIndex();

            //cout << "Image index " << imgIdx << endl;

            Pt2dr _min(DBL_MAX, DBL_MAX);
            Pt2dr _max;

            for (unsigned int bK=0; bK < region->triangles.size(); ++bK)
            {
                int triIdx = region->triangles[bK];
                cTriangle * Triangle = myMesh.getTriangle(triIdx);

                ElCamera * Cam = ListCam[imgIdx];

                vector <Pt3dr> Vertex;
                Triangle->getVertexes(Vertex);

                Pt2dr Pt1 = Cam->R3toF2(Vertex[0]);             //projection des sommets du triangle
                Pt2dr Pt2 = Cam->R3toF2(Vertex[1]);
                Pt2dr Pt3 = Cam->R3toF2(Vertex[2]);

                if (Cam->IsInZoneUtile(Pt1) || Cam->IsInZoneUtile(Pt2) || Cam->IsInZoneUtile(Pt3))
                {
                    _min = Inf(Pt1, Inf(Pt2, Inf(Pt3, _min)));
                    _max = Sup(Pt1, Sup(Pt2, Sup(Pt3, _max)));
                }
            }

            if (_min != Pt2dr(DBL_MAX, DBL_MAX)) //TODO: gerer les triangles de bord
            {
                //cout << "aK= " << aK << " img= " << imgIdx << " min, max = " << _min.x << ", " << _min.y << "  " <<  _max.x << ", " << _max.y << endl;
                region->setRect(imgIdx, round_down(_min), round_up(_max));
                //cout << "region min, max = " << region->P0() << " "  << region->P1() << endl;
            }
            else
            {
                regions[aK] = regions.back();
                regions.pop_back();
                //cout << "removing region " << aK << endl;
                aK--;
            }
        }

        cout << endl;
        cout <<"***************************Packing textures**************************"<<endl;
        cout << endl;

        const int nRegions = (int)regions.size();
        tp->setTextureCount(nRegions);

        for (int aK=0; aK < nRegions; ++aK)
        {
            Pt2di sz = regions[aK].sz();
            //cout << "aK= " << aK << " width - height " << sz.x << " " <<  sz.y << endl;
            tp->addTexture(sz.x, sz.y);
        }

        int width, height;
        int unused_area = tp->packTextures(width, height, false, false);

        cout << "Packed width-height " << width << " " << height << endl;
        printf("Unused_area : %d pixels = %1.2f %%\n", unused_area, (float) 100.f * unused_area / (width*height));

        float Scale = (float) aTextMaxSize / ElMax(width, height) ;

        if (Scale > 1.f) Scale = 1.f;

        printf("Scaling factor = %1.2f\n", Scale);
        if (Scale < 0.25f) cout << "Warning: scaling factor too low, try using higher texture size (Parameter Sz)" << endl;

        int final_width  = round_up(width * Scale);
        int final_height = round_up(height * Scale);

        cout << "Final width-height " << final_width << " " << final_height << endl;

        cout << endl;
        cout <<"***************************Writing texture***************************"<<endl;
        cout << endl;

        Tiff_Im  TiffOut = (aPhI == Tiff_Im::RGBPalette)  ?
                Tiff_Im
                (
                    aNameOut.c_str(),
                    Pt2di( final_width, final_height ),
                    GenIm::u_int1,
                    Tiff_Im::No_Compr,
                    Tiff_Im::RGB
                ) :
                Tiff_Im
                (
                    aNameOut.c_str(),
                    Pt2di( final_width, final_height ),
                    GenIm::u_int1,
                    Tiff_Im::No_Compr,
                    aPhI
                );

        for (int aK=0; aK< nRegions; aK++)
        {
            cTextureBox2d *region = &(regions[aK]);
            Pt2di p0 = region->P0();

            int x, y, w, h;
            bool rotated = tp->getTextureLocation(aK, x, y, w, h);

            //cout << "Texture " << aK << " at position " << x << ", " << y << " and rotated " << rotated << " width, height = " << w << " " << h << endl;

            int x_scaled = round_ni(x * Scale);
            int y_scaled = round_ni(y * Scale);

            //cout << "image position scaled = " << x_scaled << " " << y_scaled << endl;

            int w_scaled = round_ni(w * Scale);
            int h_scaled = round_ni(h * Scale);

            //cout << "image dimension scaled = " << w_scaled << " " << h_scaled << endl;

            int p0x_scaled = round_ni(p0.x * Scale);
            int p0y_scaled = round_ni(p0.y * Scale);

            Pt2di p0_scaled(p0x_scaled, p0y_scaled);

            Pt2di xy_scaled(x_scaled, y_scaled);
            Pt2di wh_scaled(w_scaled, h_scaled);

            int imgIdx = region->imgIdx;


            Fonc_Num aF0 = aVT[imgIdx].in_proj()  * (final_ZBufIm[imgIdx].in_proj()!=defValZBuf);

            if (rotated)
            {
                vector<Im2DGen *> aVOutInit = TiffOut.VecOfIm(Pt2di(h_scaled, w_scaled));

                Fonc_Num Fonc = StdFoncChScale(aF0,Pt2dr(p0),Pt2dr(1.f/Scale,1.f/Scale));
                Fonc = Max(0,Min(255,Fonc));
                //TODO: Si ce n'est pas une image sur 8 Bits, il est plus propre de lire les bornes avant de faire le max min
                //Fonc_Num Tronque(GenIm::type_el,Fonc_Num);

                ELISE_COPY
                (
                     aVOutInit[0]->all_pts(),
                     Fonc,
                     StdOut(aVOutInit)
                );
                //erreur : segfault avec Sz=4096 Scale=0.24

                vector<Im2DGen *>   aVOutRotate;
                for (int aK=0 ; aK<int(aVOutInit.size()) ; aK++)
                     aVOutRotate.push_back(aVOutInit[aK]->ImRotate(3));

                ELISE_COPY
                (
                    rectangle(xy_scaled, xy_scaled + wh_scaled),
                    trans(StdInput(aVOutRotate), -xy_scaled),
                    TiffOut.out()
                );

                region->setTransfo(xy_scaled, rotated);
            }
            else
            {
                Pt2di tr = p0_scaled - xy_scaled;

                region->setTransfo(-tr, rotated);

                Fonc_Num aF = aF0;
                while (aF.dimf_out() < aNbCh)
                    aF = Virgule(aF0,aF);
                aF = StdFoncChScale(aF,Pt2dr(), Pt2dr(1.f/Scale,1.f/Scale));

                ELISE_COPY
                (
                    rectangle(xy_scaled, xy_scaled + wh_scaled),
                    trans(aF, tr),
                    TiffOut.out()
                );
            }
        }

        releaseTexturePacker(tp);

        cout << endl;
        cout <<"********************Computing texture coordinates********************"<<endl;
        cout << endl;


        Pt2dr p0_scaled, p1_scaled;

        int cptTmp =0;
        vector < cTextureBox2d >::const_iterator it = regions.begin();
        for (; it != regions.end(); ++it, cptTmp++)
        {
            float tx = it->translation.x;
            float ty = it->translation.y;

            if (it->isRotated)
            {
                p0_scaled = Pt2dr(it->P0())*Scale;
                p1_scaled = Pt2dr(it->P1())*Scale;
            }

            //cout << "rotated " << it->isRotated << " translation = " << it->translation << " p0 = " <<p0_scaled << " p1 = " << p1_scaled << endl;

            //cout << "nb Triangles = " << it->sz() << endl;

            for (unsigned int bK=0; bK < it->triangles.size();++bK)
            {
                int triIdx = it->triangles[bK];

                cTriangle *Triangle = myMesh.getTriangle(triIdx);

                int idx = Triangle->getBestImgIndex();                //Liaison avec l'image correspondante

                //cout << "image pour le triangle " << i << " = " << idx << endl;

                if (idx != valDef)
                {
                    CamStenope *Cam = ListCam[idx];

                    vector <Pt3dr> Vertex;
                    Triangle->getVertexes(Vertex);

                    Pt2dr Pt1 = Cam->R3toF2(Vertex[0]);             //projection des sommets du triangle
                    Pt2dr Pt2 = Cam->R3toF2(Vertex[1]);
                    Pt2dr Pt3 = Cam->R3toF2(Vertex[2]);

                    //debug
                    vector <int> vIndexes;
                    Triangle->getVertexesIndexes(vIndexes);

                    if ((vIndexes[0] == 14473) && (vIndexes[1] == 14473) && (vIndexes[2]==15050))
                    {
                        cout << "idx du triangle = " << triIdx << " region = " << cptTmp<< " bK = " << bK << " rotated " << it->isRotated << endl;
                        cout << "nb triangles dans la region = "<< it->triangles.size()<< endl;
                        cout << "p0 = "<< it->P0() << endl;
                        cout << "p1 = "<< it->P1() << endl;
                        cout << "Pt1 = " << Pt1 << endl;
                        cout << "Pt2 = " << Pt2 << endl;
                        cout << "Pt3 = " << Pt3 << endl;
                        cout << "tx = " << tx << endl;
                        cout << "ty = " << ty << endl;
                        cout << "idcam = " << Cam->IdentCam() << endl;
                    }
                    //end debug

                    if (Cam->IsInZoneUtile(Pt1) || Cam->IsInZoneUtile(Pt2) || Cam->IsInZoneUtile(Pt3))
                    {
                        Pt2dr Pt1s = Pt1*Scale;
                        Pt2dr Pt2s = Pt2*Scale;
                        Pt2dr Pt3s = Pt3*Scale;

                        Pt2dr P1, P2, P3;

                        if(it->isRotated)
                        {
                            P1.x = (Pt1s.y + tx - p0_scaled.y) / final_width;
                            P2.x = (Pt2s.y + tx - p0_scaled.y) / final_width;
                            P3.x = (Pt3s.y + tx - p0_scaled.y) / final_width;

                            P1.y = 1.f - (-Pt1s.x + ty + p1_scaled.x) / final_height;
                            P2.y = 1.f - (-Pt2s.x + ty + p1_scaled.x) / final_height;
                            P3.y = 1.f - (-Pt3s.x + ty + p1_scaled.x) / final_height;

                           /* if (bK== 0)
                            {
                                cout << "Pt1 = " << Pt1 << endl;
                                cout << "Pt2 = " << Pt2 << endl;
                                cout << "Pt3 = " << Pt3 << endl;
                            }*/
                        }
                        else
                        {
                            P1.x = (Pt1s.x+tx) / final_width;
                            P2.x = (Pt2s.x+tx) / final_width;
                            P3.x = (Pt3s.x+tx) / final_width;

                            P1.y = 1.f - (Pt1s.y+ty) / final_height;
                            P2.y = 1.f - (Pt2s.y+ty) / final_height;
                            P3.y = 1.f - (Pt3s.y+ty) / final_height;
                        }

                        //if ((P1.x >=0.f) && (P1.x <= 1.f) && (P2.x >=0.f) && (P2.y <= 1.f) && (P3.x >=0.f) && (P3.y <= 1.f))
                            Triangle->setTextureCoordinates(P1, P2, P3);
                        /*else
                        {
                            myMesh.removeTriangle(*Triangle);
                            updateIndex(triIdx, regions);
                        }*/
                    }
                    /*else
                    {
                        myMesh.removeTriangle(*Triangle);
                        updateIndex(triIdx, regions);
                    }*/
                }
            }
        }
    }
    else if (aMode == "Basic")
    {
        vector <Pt2dr> TabCoor;

        int aNbLine = round_up(sqrt(double(aVT.size())));
        int aNbCol = round_up(aVT.size()/double(aNbLine));

        cout << aNbLine << " rows and "  << aNbCol <<" columns texture, with "<< aVT.size() <<" images. "<< endl;
        cout << endl;

        int full_width  = aSzMax.x * aNbCol;
        int full_height = aSzMax.y * aNbLine;

        float Scale = (float) aTextMaxSize / ElMax(full_width, full_height) ;

        if (Scale > 1.f) Scale = 1.f;

        cout << "Scaling factor = " << Scale << endl;

        int final_width  = round_up(full_width * Scale);
        int final_height = round_up(full_height * Scale);

        Pt2di aSz ( final_width, final_height );

        //cout << "SZ = " << aSz << " :: " << aNbCol << " X " << aNbLine  << "\n";

        Tiff_Im  TiffOut = (aPhI == Tiff_Im::RGBPalette)  ?
                Tiff_Im
                (
                    aNameOut.c_str(),
                    aSz,
                    GenIm::u_int1,
                    Tiff_Im::No_Compr,
                    Tiff_Im::RGB
                ) :
                Tiff_Im
                (
                    aNameOut.c_str(),
                    aSz,
                    GenIm::u_int1,
                    Tiff_Im::No_Compr,
                    aPhI
                );

        const int nImg = (int)aVT.size();
        for (int aK=0 ; aK< nImg ; aK++)
        {
            Pt2di ptK(aK % aNbCol, aK / aNbCol);

            //cout << "WRITE " << aVT[aK].name() << "\n";

            Pt2di aP0 (
                        (ptK.x*aSz.x) / aNbCol,
                        (ptK.y*aSz.y) / aNbLine
                        );

            Pt2di aP1 (
                        ((ptK.x+1)*aSz.x) / aNbCol,
                        ((ptK.y+1)*aSz.y) / aNbLine
                        );

            Fonc_Num aF0 = aVT[aK].in_proj() * (final_ZBufIm[aK].in_proj()!=defValZBuf);
            Fonc_Num aF = aF0;
            while (aF.dimf_out() < aNbCh)
                aF = Virgule(aF0,aF);
            aF = StdFoncChScale(aF,Pt2dr(-aP0.x,-aP0.y)/Scale, Pt2dr(1.f/Scale,1.f/Scale));

            ELISE_COPY
                    (
                        rectangle(aP0,aP1),
                        aF ,
                        TiffOut.out()
                        );

            Pt2dr Coord = ptK.mcbyc(aVT[aK].sz())*Scale;

            TabCoor.push_back(Coord);

            /*   cout<<"Ligne : "<<ptK.y+1 << " Colonne : "<<ptK.x+1<<endl;
            cout<<"Position : "<< Coord.x <<" " << Coord.y <<endl;
            cout<<"Nombre d'images traitees : "<<aK+1<<"/"<<aVT.size()<<endl;
            cout<<endl;*/
        }

        cout << endl;
        cout <<"********************Computing texture coordinates********************"<<endl;
        cout << endl;

        //cout << "myMesh.getFacesNumber()= "<< myMesh.getFacesNumber() << endl;
        for(int i=0 ; i< myMesh.getFacesNumber() ; i++)                          //Ecriture des triangles
        {
            cTriangle * Triangle = myMesh.getTriangle(i);

            int idx = Triangle->getBestImgIndex();                //Liaison avec l'image correspondante

            //cout << "image pour le triangle " << i << " = " << idx << endl;

            if (idx != valDef)
            {
                CamStenope * Cam = ListCam[idx];

                vector <Pt3dr> Vertex;
                Triangle->getVertexes(Vertex);

                Pt2dr Pt1 = Cam->R3toF2(Vertex[0]);             //projection des sommets du triangle
                Pt2dr Pt2 = Cam->R3toF2(Vertex[1]);
                Pt2dr Pt3 = Cam->R3toF2(Vertex[2]);

                if (Cam->IsInZoneUtile(Pt1) || Cam->IsInZoneUtile(Pt2) || Cam->IsInZoneUtile(Pt3))
                {
                    Pt2dr PtTemp = TabCoor[idx];

                    //cout << "PtTemp = " <<  PtTemp << endl;

                    Pt2dr P1, P2, P3;

                    P1.x = ((float)(Pt1.x*Scale)+PtTemp.x) / final_width;
                    P2.x = ((float)(Pt2.x*Scale)+PtTemp.x) / final_width;
                    P3.x = ((float)(Pt3.x*Scale)+PtTemp.x) / final_width;

                    P1.y = 1.f - ((float)(Pt1.y*Scale)+PtTemp.y) / final_height;
                    P2.y = 1.f - ((float)(Pt2.y*Scale)+PtTemp.y) / final_height;
                    P3.y = 1.f - ((float)(Pt3.y*Scale)+PtTemp.y) / final_height;

                    Triangle->setTextureCoordinates(P1, P2, P3);
                }
                else
                {
                    myMesh.removeTriangle(*Triangle);
                    i--;
                }
            }
        }
    }

    cout << endl;
    cout <<"***********************Converting texture file***********************"<<endl;
    cout << endl;

    string aCom =  g_externalToolHandler.get( "convert" ).callName() + string(" -quality ") + st.str() + " "
            + aNameOut + " " + textureName;

    if (debug) cout << "COM= " << aCom << endl;

    system_call(aCom.c_str());

    aCom = string(SYS_RM) + " " + aNameOut;
    system_call(aCom.c_str());

    cout << endl;
    cout <<"**************************Writing ply file***************************"<<endl;
    cout << endl;

    myMesh.write(aOut, aBin, textureName);

    cout <<"********************************Done*********************************"<<endl;
    cout << endl;

    return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/

