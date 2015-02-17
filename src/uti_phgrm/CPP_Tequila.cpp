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

/*#if ELISE_Darwin
    #include <OpenGL/gl.h>
#else
    #include <GL/gl.h>
#endif*/

bool debug = false;
float defValZBuf = 1e9;

int Tequila_main(int argc,char ** argv)
{
    int maxTextureSize, texture_units;
    maxTextureSize = texture_units = 0;
    /*glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);

    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &texture_units);

    printf("Max texture size / Max texture units: %d / %d\n", maxTextureSize, texture_units);*/

    if (maxTextureSize == 0) maxTextureSize = 8192;

    std::string aDir, aPat, aFullName, aOri, aPly;
    std::string aOut, aTextOut;
    int aTextMaxSize = maxTextureSize;
    int aZBuffSSEch = 1;
    int aJPGcomp = 70;
    double aAngleMin = 60.f;
    bool aBin = true;
    bool aModeClassic = false;

    std::stringstream sst;
    sst << aTextMaxSize;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullName,"Full Name (Dir+Pat)",eSAM_IsPatFile)
                            << EAMC(aOri,"Orientation path",eSAM_IsExistDirOri)
                            << EAMC(aPly,"Ply file", eSAM_IsExistFile),
                LArgMain()  << EAM(aOut,"Out",true,"Textured mesh name (def=plyName+ _textured.ply)")
                            << EAM(aBin,"Bin",true,"Write binary ply (def=true)")
                            << EAM(aTextOut,"Texture",true,"Texture name (def=plyName + _UVtexture.jpg)")
                            << EAM(aTextMaxSize,"Sz",true,"Texture max size (def="+ sst.str() +")")
                            << EAM(aZBuffSSEch,"Scale", true, "Z-buffer downscale factor (def=1)",eSAM_InternalUse)
                            << EAM(aJPGcomp, "QUAL", true, "jpeg compression quality (def=70)")
                            << EAM(aAngleMin, "Angle", true, "Threshold angle, in degree, between triangle normal and image viewing direction (def=60)")
                            << EAM(aModeClassic,"Mode", true, "Mode (def = false)")
             );

    if (MMVisualMode) return EXIT_SUCCESS;

    if (aTextMaxSize > maxTextureSize)
    {
        std::stringstream sst2;
        sst2 << maxTextureSize;
        cout << "Warning: trying to write texture higher than GL_MAX_TEXTURE_SIZE (" + sst2.str() + ")";
        //return;
    }

    SplitDirAndFile(aDir,aPat,aFullName);

    if (!EAMIsInit(&aOut)) aOut = StdPrefix(aPly) + "_textured.ply";
    if (!EAMIsInit(&aTextOut)) aTextOut = StdPrefix(aPly) + "_UVtexture.tif";

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    std::list<std::string>  aLS = aICNM->StdGetListOfFile(aPat);

    // If the users enters Ori-MyOrientation/, it will be corrected into MyOrientation
    StdCorrecNameOrient(aOri,aDir);

    std::vector<CamStenope*> ListCam;

    cout << endl;
    for (std::list<std::string>::const_iterator itS=aLS.begin(); itS!=aLS.end() ; itS++)
    {
        std::string NOri=aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aOri,*itS,true);

        ListCam.push_back(CamOrientGenFromFile(NOri,aICNM));

        cout<<"Image "<<*itS<<", with ori : "<< NOri <<endl;
    }

    cout<<endl;
    cout<<"**************************Reading ply file***************************"<<endl;
    cout<<endl;

    cMesh myMesh(aPly);

    printf("Vertex number : %d - faces number : %d \n\n", myMesh.getVertexNumber(), myMesh.getFacesNumber());

    cout<<"*************************Computing Z-Buffer**************************"<< endl;
    cout<< endl;

    vector <cZBuf> aZBuffers;

    std::list<std::string>::const_iterator itS=aLS.begin();
    for(unsigned int aK=0 ; aK<ListCam.size() ; aK++, itS++)
    {
        cout << "Z-buffer " << aK+1 << "/" << ListCam.size() << endl;

        cZBuf aZBuffer(ListCam[aK]->Sz(), defValZBuf, aZBuffSSEch);

        aZBuffer.BasculerUnMaillage(myMesh, *ListCam[aK]);

        if (debug)
        {
            Im2D_REAL4 res = aZBuffer.get();

            //conversion du zBuffer en 8 bits
            Pt2di sz = res.sz(); //aZBuffer.Sz();
            Im2D_U_INT1 Converted(sz.x, sz.y);
            REAL min = FLT_MAX;
            REAL max = 0.f;
            for (int cK=0; cK < sz.x;++cK)
            {
                for (int bK=0; bK < sz.y;++bK)
                {
                    REAL val = res.GetR(Pt2di(cK,bK));

                    if (val != defValZBuf)
                    {
                        max = ElMax(val, max);
                        min = ElMin(val, min);
                    }
                }
            }

            printf ("Min, max depth = %4.2f %4.2f\n", min, max );

            for (int cK=0; cK < sz.x;++cK)
                for (int bK=0; bK < sz.y;++bK)
                    Converted.SetI(Pt2di(cK,bK),(int)((res.GetR(Pt2di(cK,bK))-min) *255.f/(max-min)));

            std::stringstream ss  ;
            ss << aK;
            string filename = StdPrefix(*itS) + "_zbuf" + ss.str() + ".tif";
            printf ("Saving %s\n", filename.c_str());
            Tiff_Im::CreateFromIm(Converted, filename);
            printf ("Done\n");

            //image des labels
            Im2D_INT4 Labels = aZBuffer.getIndexImage();

            //conversion de l'img de label en 8 bits
            Im2D_U_INT1 LConverted(sz.x, sz.y);
            int lmin = INT_MAX;
            int lmax = 0;
            for (int cK=0; cK < sz.x;++cK)
            {
                for (int bK=0; bK < sz.y;++bK)
                {
                    int val = Labels.GetI(Pt2di(cK,bK));

                    if (val != INT_MAX)
                    {
                        lmax = ElMax(val, lmax);
                        lmin = ElMin(val, lmin);
                    }
                }
            }

            for (int cK=0; cK < sz.x;++cK)
                for (int bK=0; bK < sz.y;++bK)
                    LConverted.SetI(Pt2di(cK,bK),(int)((Labels.GetI(Pt2di(cK,bK))-lmin) *255.f/(lmax-lmin)));

            filename = StdPrefix(*itS) + "_label" + ss.str() + ".tif";
            printf ("Saving %s\n", filename.c_str());
            Tiff_Im::CreateFromIm(LConverted, filename);
            printf ("Done\n");
        }

        aZBuffers.push_back(aZBuffer);
    }

    cout << endl;
    cout << "************************Choosing best image**************************" << endl;
    cout << endl;

    std::set <int> index; //liste des index de cameras utilisees
    int valDef = cTriangle::getDefTextureImgIndex();

    float threshold =  cos(PI*(1.f - aAngleMin/180.f)); //angle min = cos(180 - 60) = -0.5
    //cout << "threshold=" << threshold << endl;

    for(int i=0 ; i < myMesh.getFacesNumber(); i++)                            //Pour un triangle
    {
        float PScalcur = threshold;

        int idx = valDef;
        for(unsigned int j=0 ; j<ListCam.size() ; j++) // on teste toutes les CamStenope
        {
            vector <unsigned int> vTri = aZBuffers[j].getVisibleTrianglesIndexes();

            if (std::find(vTri.begin(),vTri.end(), i) != vTri.end())
            {
                cTriangle * Triangle = myMesh.getTriangle(i);

                double PScalnew = scal(Triangle->getNormale(true), ListCam[j]->DirK());
                //cout << "scal= " << PScalnew << endl;
                if(PScalnew<PScalcur)        //On garde celle pour laquelle le PS est le plus fort
                {
                    PScalcur = PScalnew;

                    Triangle->setTextureImgIndex(j);
                    idx = j;
                }
            }
        }
        if (idx != valDef) index.insert(idx);
    }

    cout << "Selected images / total : " << index.size() << " / " << aLS.size() << endl;

    cout << endl;
    cout <<"********************Filtering border triangles***********************"<<endl;
    cout << endl;

    int maxIter = 20; //TODO: a mettre en parametre ?
    int iter = 0;
    bool cond = true;

    while (cond && iter < maxIter)
    {
        //cout << "myMesh.getFacesNb " << myMesh.getFacesNumber() << endl;
        myMesh.clean();

        //cout << "myMesh.getFacesNb " << myMesh.getFacesNumber() << endl;

        iter++;
        cout << "round " << iter << endl;

        if (iter!= maxIter)
        {
            cond = false;
            for (int aK=0; aK< myMesh.getFacesNumber();++aK)
            {
                cTriangle * triangle = myMesh.getTriangle(aK);
                if (triangle->getEdgesNumber() < 3 && !triangle->isTextured())
                {
                    cond = true;
                    break;
                }
            }
        }
    }

    printf("\nVertex number : %d - faces number : %d \n", myMesh.getVertexNumber(), myMesh.getFacesNumber());

    Pt2di aSzMax;
    vector <Tiff_Im> aVT;     //Vecteur contenant les images
    int aNbCh = 0;

    vector <Im2D_REAL4> final_ZBufIm;

    for (std::set<int>::const_iterator it=index.begin(); it!=index.end(); ++it)
    {
        int bK=0;
        for (std::list<std::string>::const_iterator itS=aLS.begin(); itS!=aLS.end() ; itS++, bK++)
        {
            if (*it == bK)
            {
                aVT.push_back(Tiff_Im::StdConvGen(aDir+*itS,-1,false,true));
                final_ZBufIm.push_back(aZBuffers[*it].get());
                aSzMax.SetSup(aVT.back().sz());
                aNbCh = ElMax(aNbCh,aVT.back().nb_chan());
                break;
            }
        }
    }

    if (!aModeClassic)
    {

        cout << endl;
        cout <<"***********************Getting adjacent triangles***********************"<<endl;
        cout << endl;

        std::vector < cTextRect > regions = myMesh.getRegions();
        cout << "nb regions = " << regions.size() << endl;

        TEXTURE_PACKER::TexturePacker *tp = TEXTURE_PACKER::createTexturePacker();

        int nbTriangles = 0;
        for (unsigned int aK=0; aK < regions.size();++aK)
        {
            cout << "region " << aK << " nb triangles = " << regions[aK].triangles.size() << endl;
            //Calcul de la zone correspondante dans l'image

            int triIdx = regions[aK].triangles[0];
            cTriangle * Tri = myMesh.getTriangle(triIdx);
            int imgIdx = Tri->getTextureImgIndex();

            cout << "Image index " << imgIdx << endl;

            Pt2dr _min(DBL_MAX, DBL_MAX);
            Pt2dr _max;

            for (unsigned int bK=0; bK < regions[aK].triangles.size(); ++bK)
            {
                int triIdx = regions[aK].triangles[bK];
                cTriangle * Triangle = myMesh.getTriangle(triIdx);

                ElCamera * Cam = ListCam[imgIdx];

                vector <Pt3dr> Vertex;
                Triangle->getVertexes(Vertex);

                Pt2dr Pt1 = Cam->R3toF2(Vertex[0]);             //projection des sommets du triangle
                Pt2dr Pt2 = Cam->R3toF2(Vertex[1]);
                Pt2dr Pt3 = Cam->R3toF2(Vertex[2]);

                if (Cam->IsInZoneUtile(Pt1) && Cam->IsInZoneUtile(Pt2) && Cam->IsInZoneUtile(Pt3))
                {
                    _min = Inf(Pt1, _min);
                    _min = Inf(Pt2, _min);
                    _min = Inf(Pt3, _min);

                    _max = Sup(Pt1, _max);
                    _max = Sup(Pt2, _max);
                    _max = Sup(Pt3, _max);
                }
            }

            if (_min != Pt2dr(DBL_MAX, DBL_MAX)) //TODO: gerer les triangles de bord
            {
                nbTriangles += regions[aK].triangles.size();
                //cout << "min, max = " << _min.x << ", " << _min.y << "  " <<  _max.x << ", " << _max.y << endl;
                regions[aK].setRect(imgIdx, round_down(_min), round_up(_max));

            }
            else
            {
                //cout << "removing region " << aK << endl;
                regions.erase(std::remove(regions.begin(), regions.end(), regions[aK]), regions.end());
                aK--;
            }
        }

        cout << endl;
        cout <<"***********************Packing textures***********************"<<endl;
        cout << endl;

        cout << "Triangles nb = " << nbTriangles << endl;
        tp->setTextureCount(regions.size());

        for (unsigned int aK=0; aK < regions.size(); ++aK)
        {
            Pt2di sz = regions[aK].size();
            //cout << "width - height " << sz.x << " " <<  sz.y << endl;
            tp->addTexture(sz.x, sz.y);
        }

        int width, height;
        int unused_area = tp->packTextures(width, height, false, false);

        cout << "final width-height " << width << " " << height << endl;
        cout << "unused_area : " << unused_area << " = " << (float) unused_area/ (width*height) << "%" << endl;

        cout << endl;
        cout <<"**************************Writing texture**************************"<<endl;
        cout << endl;

        Tiff_Im  nFileRes
                (
                    aTextOut.c_str(),
                    Pt2di( width, height ),
                    GenIm::u_int1,
                    Tiff_Im::No_Compr,
                    Tiff_Im::RGB
                );

        for (unsigned int aK=0; aK< regions.size(); aK++)
        {
            int imgIdx = regions[aK].imgIdx;

            int x, y, w, h;
            bool rotated = tp->getTextureLocation(aK, x, y, w, h);

            //cout << "position dans l'image " << imgIdx << " = " << regions[aK].p0.x << " " << regions[aK].p0.y << endl;

            Pt2di tr =  regions[aK].p0 - Pt2di(x,y);
            regions[aK].translation = tr;
            regions[aK].rotation = rotated;

            //TODO: prendre en compte le facteur de sous-ech sur final_ZBufIm
            Fonc_Num aF0 = aVT[imgIdx].in_proj() * (final_ZBufIm[imgIdx].in_proj()!=defValZBuf);

            if (rotated)
            {
                // Can only handle RLE mode for File-Images

                /* ELISE_COPY
                 (
                     rectangle(Pt2di(x,y),Pt2di(x+w,y+h)),
                     aF0 [Virgule(FY+Q0,Q1-FX)],
                     nFileRes.out()
                 );*/
            }
            else
            {
                ELISE_COPY
                (
                    rectangle(Pt2di(x,y),Pt2di(x+w,y+h)),
                    trans(aF0, tr),
                    nFileRes.out()
                );
            }

            cout << "Texture " << aK << " at position " << x << ", " << y << " and rotated " << rotated << " width, height = " << w << " " << h << endl;
        }

        releaseTexturePacker(tp);

        std::string newName = StdPrefix(aTextOut) + ".jpg ";
        std::stringstream st  ;
        st << aJPGcomp;

        std::string aCom = g_externalToolHandler.get( "convert" ).callName() + std::string(" -quality ") + st.str() + " "
                + aTextOut + " " + newName;

        //cout << "COM= " << aCom << endl;

        system_call(aCom.c_str());

        aCom = std::string(SYS_RM) + " " + aTextOut;
        system_call(aCom.c_str());

        cout << endl;
        cout <<"**************************Writing ply file***************************"<<endl;
        cout <<endl;

        string mode = aBin ? "wb" : "w";
        string aBinSpec = MSBF_PROCESSOR() ? "binary_big_endian":"binary_little_endian" ;

        FILE * PlyOut = FopenNN(aOut,mode, "UV Mapping");         //Ecriture du header
        fprintf(PlyOut,"ply\n");
        fprintf(PlyOut,"format %s 1.0\n",aBin?aBinSpec.c_str():"ascii");
        fprintf(PlyOut,"comment UV Mapping generated\n");
        fprintf(PlyOut,"comment TextureFile %s\n", newName.c_str());
        fprintf(PlyOut,"element vertex %i\n",myMesh.getVertexNumber());
        fprintf(PlyOut,"property float x\n");
        fprintf(PlyOut,"property float y\n");
        fprintf(PlyOut,"property float z\n");
        fprintf(PlyOut,"element face %i\n",nbTriangles);
        fprintf(PlyOut,"property list uchar int vertex_indices\n");
        fprintf(PlyOut,"property list uchar float texcoord\n");
        fprintf(PlyOut,"end_header\n");

        Pt3dr pt;
        int t1, t2, t3;
        cTriangle *Triangle;

        CamStenope *Cam;
        int idx;

        if (aBin)
        {
            for(int aK=0 ; aK< myMesh.getVertexNumber() ; aK++) //Ecriture des vertex
            {
                pt = myMesh.getVertex(aK);

                WriteType(PlyOut,float(pt.x));
                WriteType(PlyOut,float(pt.y));
                WriteType(PlyOut,float(pt.z));
            }
        }
        else
        {
            for(int aK=0 ; aK< myMesh.getVertexNumber() ; aK++) //Ecriture des vertex
            {
                pt = myMesh.getVertex(aK);
                fprintf(PlyOut,"%.7f %.7f %.7f\n",pt.x,pt.y,pt.z);
            }
        }

        //TODO
        float Scale = 1.f;

        for (unsigned int aK=0; aK < regions.size(); ++aK)
        {
            Pt2di PtTemp = -regions[aK].translation;
            bool rotat = regions[aK].rotation;

            //cout << "nb Triangles = " << regions[aK].size() << endl;

                for (unsigned int bK=0; bK < regions[aK].triangles.size();++bK)
                {
                    int triIdx = regions[aK].triangles[bK];

                    Triangle = myMesh.getTriangle(triIdx);
                    Triangle->getVertexesIndexes(t1,t2,t3);              //On recupere les sommets de chaque triangle

                    idx = Triangle->getTextureImgIndex();                //Liaison avec l'image correspondante

                    //cout << "image pour le triangle " << i << " = " << idx << endl;

                    if (idx != valDef)
                    {
                        Cam = ListCam[idx];

                        vector <Pt3dr> Vertex;
                        Triangle->getVertexes(Vertex);

                        Pt2dr Pt1 = Cam->R3toF2(Vertex[0]);             //projection des sommets du triangle
                        Pt2dr Pt2 = Cam->R3toF2(Vertex[1]);
                        Pt2dr Pt3 = Cam->R3toF2(Vertex[2]);

                        if (Cam->IsInZoneUtile(Pt1) && Cam->IsInZoneUtile(Pt2) && Cam->IsInZoneUtile(Pt3))
                        {
                            /*cout << "Pt1= " << Pt1.x << " " << Pt1.y << endl;
                        cout << "Pt2= " << Pt2.x << " " << Pt2.y << endl;
                        cout << "Pt3= " << Pt3.x << " " << Pt3.y << endl;*/

                            //cout << "idx= " << idx << endl;

                            //cout << "PtTemp = " <<  PtTemp << endl;
                            if(rotat)
                            {
                                int rwidth = regions[aK].width();

                                float x1_tmp = Pt1.x;
                                float x2_tmp = Pt2.x;
                                float x3_tmp = Pt3.x;

                                Pt1.x = Pt1.y;
                                Pt2.x = Pt2.y;
                                Pt3.x = Pt3.y;

                                Pt1.y = rwidth - x1_tmp;
                                Pt2.y = rwidth - x2_tmp;
                                Pt3.y = rwidth - x3_tmp;
                            }

                            float Pt1x=((float)(Pt1.x*Scale)+PtTemp.x) / width;
                            float Pt1y= 1.f - ((float)(Pt1.y*Scale)+PtTemp.y) / height;
                            float Pt2x=((float)(Pt2.x*Scale)+PtTemp.x) / width;
                            float Pt2y= 1.f - ((float)(Pt2.y*Scale)+PtTemp.y) / height;
                            float Pt3x=((float)(Pt3.x*Scale)+PtTemp.x) / width;
                            float Pt3y= 1.f - ((float)(Pt3.y*Scale)+PtTemp.y) / height;

                            if (aBin)
                            {
                                WriteType(PlyOut,(unsigned char)3);
                                WriteType(PlyOut,t1);
                                WriteType(PlyOut,t2);
                                WriteType(PlyOut,t3);
                                WriteType(PlyOut,(unsigned char)6);
                                WriteType(PlyOut,Pt1x);
                                WriteType(PlyOut,Pt1y);
                                WriteType(PlyOut,Pt2x);
                                WriteType(PlyOut,Pt2y);
                                WriteType(PlyOut,Pt3x);
                                WriteType(PlyOut,Pt3y);
                            }
                            else
                            {
                                fprintf(PlyOut,"3 %i %i %i ",t1,t2,t3);
                                fprintf(PlyOut,"6 %f %f %f %f %f %f \n",Pt1x,Pt1y,Pt2x,Pt2y,Pt3x,Pt3y);
                            }
                        }
                        else
                        {
                            //cout << "HORS IMG" << endl;
                            if (aBin)
                            {
                                WriteType(PlyOut,(unsigned char)3);
                                WriteType(PlyOut,t1);
                                WriteType(PlyOut,t2);
                                WriteType(PlyOut,t3);
                                WriteType(PlyOut,(unsigned char)6);
                                WriteType(PlyOut,float(0));
                                WriteType(PlyOut,float(0));
                                WriteType(PlyOut,float(0));
                                WriteType(PlyOut,float(0));
                                WriteType(PlyOut,float(0));
                                WriteType(PlyOut,float(0));
                            }
                            else
                            {
                                fprintf(PlyOut,"3 %i %i %i ",t1,t2,t3);
                                fprintf(PlyOut,"6 %d %d %d %d %d %d \n",0,0,0,0,0,0);
                            }
                        }
                    }
                    else
                    {
                        if (aBin)
                        {
                            WriteType(PlyOut,(unsigned char)3);
                            WriteType(PlyOut,t1);
                            WriteType(PlyOut,t2);
                            WriteType(PlyOut,t3);
                            WriteType(PlyOut,(unsigned char)6);
                            WriteType(PlyOut,float(0));
                            WriteType(PlyOut,float(0));
                            WriteType(PlyOut,float(0));
                            WriteType(PlyOut,float(0));
                            WriteType(PlyOut,float(0));
                            WriteType(PlyOut,float(0));
                        }
                        else
                        {
                            fprintf(PlyOut,"3 %i %i %i ",t1,t2,t3);
                            fprintf(PlyOut,"6 %d %d %d %d %d %d \n",0,0,0,0,0,0);
                        }
                    }
                }

        }

        /* // plus grande texture en premier
          while(regions.size())
        {
            unsigned int sz = 0;
            int biggest = -1;
            for (unsigned int aK=0; aK < regions.size();++aK)
            {
                if (regions[aK].size() > sz)
                {
                    sz = regions[aK].size();
                    biggest = aK;
                }
            }

            cout << "Biggest region= " << biggest << " with " << regions[biggest].size() << " triangles " << endl;

            //Calcul de la zone correspondante dans l'image

            int triIdx = regions[biggest][0];
            cTriangle * Tri = myMesh.getTriangle(triIdx);
            int imgIdx = Tri->getTextureImgIndex();

            cout << "Image index " << imgIdx << endl;

            Pt2dr _min(DBL_MAX, DBL_MAX);
            Pt2dr _max;

            for (unsigned int aK=0; aK < regions[biggest].size(); ++aK)
            {
                int triIdx = regions[biggest][aK];
                cTriangle * Triangle = myMesh.getTriangle(triIdx);

                ElCamera * Cam = ListCam[imgIdx];

                vector <Pt3dr> Vertex;
                Triangle->getVertexes(Vertex);

                Pt2dr Pt1 = Cam->R3toF2(Vertex[0]);             //projection des sommets du triangle
                Pt2dr Pt2 = Cam->R3toF2(Vertex[1]);
                Pt2dr Pt3 = Cam->R3toF2(Vertex[2]);

                if (Cam->IsInZoneUtile(Pt1) && Cam->IsInZoneUtile(Pt2) && Cam->IsInZoneUtile(Pt3))
                {
                    _min = Inf(Pt1, _min);
                    _min = Inf(Pt2, _min);
                    _min = Inf(Pt3, _min);

                    _max = Sup(Pt1, _max);
                    _max = Sup(Pt2, _max);
                    _max = Sup(Pt3, _max);
                }
            }

            cout << "min, max = " << _min.x << ", " << _min.y << "  " <<  _max.x << ", " << _max.y << endl;
            vRect.push_back(TextRect(imgIdx, round_down(_min), round_up(_max)));

            regions.erase(std::remove(regions.begin(), regions.end(), regions[biggest]), regions.end());
        }*/


    }
    else //mode classic
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

        Tiff_Im::PH_INTER_TYPE aPhI = aVT[0].phot_interp();
        if (aNbCh==3)
            aPhI = Tiff_Im::RGB;

        Tiff_Im  FileRes
                (
                    aTextOut.c_str(),
                    aSz,
                    GenIm::u_int1,
                    Tiff_Im::No_Compr,
                    aPhI
                    );

        for (int aK=0 ; aK<int(aVT.size()) ; aK++)
        {
            Pt2di ptK(aK % aNbCol, aK / aNbCol);

            //std::cout << "WRITE " << aVT[aK].name() << "\n";

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
                        FileRes.out()
                        );

            Pt2dr Coord = ptK.mcbyc(aVT[aK].sz())*Scale;

            TabCoor.push_back(Coord);

            /*   cout<<"Ligne : "<<ptK.y+1 << " Colonne : "<<ptK.x+1<<endl;
            cout<<"Position : "<< Coord.x <<" " << Coord.y <<endl;
            cout<<"Nombre d'images traitees : "<<aK+1<<"/"<<aVT.size()<<endl;
            cout<<endl;*/
        }

        std::string newName = StdPrefix(aTextOut) + ".jpg ";
        std::stringstream st  ;
        st << aJPGcomp;

        std::string aCom =  g_externalToolHandler.get( "convert" ).callName() + std::string(" -quality ") + st.str() + " "
                + aTextOut + " " + newName;

        //cout << "COM= " << aCom << endl;

        system_call(aCom.c_str());

        aCom = std::string(SYS_RM) + " " + aTextOut;
        system_call(aCom.c_str());

        cout << endl;
        cout <<"**************************Writing ply file***************************"<<endl;
        cout <<endl;

        string mode = aBin ? "wb" : "w";
        string aBinSpec = MSBF_PROCESSOR() ? "binary_big_endian":"binary_little_endian" ;

        FILE * PlyOut = FopenNN(aOut,mode, "UV Mapping");         //Ecriture du header
        fprintf(PlyOut,"ply\n");
        fprintf(PlyOut,"format %s 1.0\n",aBin?aBinSpec.c_str():"ascii");
        fprintf(PlyOut,"comment UV Mapping generated\n");
        fprintf(PlyOut,"comment TextureFile %s\n", newName.c_str());
        fprintf(PlyOut,"element vertex %i\n",myMesh.getVertexNumber());
        fprintf(PlyOut,"property float x\n");
        fprintf(PlyOut,"property float y\n");
        fprintf(PlyOut,"property float z\n");
        fprintf(PlyOut,"element face %i\n",myMesh.getFacesNumber());
        fprintf(PlyOut,"property list uchar int vertex_indices\n");
        fprintf(PlyOut,"property list uchar float texcoord\n");
        fprintf(PlyOut,"end_header\n");

        Pt3dr pt;
        int t1, t2, t3;
        cTriangle *Triangle;

        CamStenope *Cam;
        int idx;

        if (aBin)
        {
            for(int aK=0 ; aK< myMesh.getVertexNumber() ; aK++) //Ecriture des vertex
            {
                pt = myMesh.getVertex(aK);

                WriteType(PlyOut,float(pt.x));
                WriteType(PlyOut,float(pt.y));
                WriteType(PlyOut,float(pt.z));
            }
        }
        else
        {
            for(int aK=0 ; aK< myMesh.getVertexNumber() ; aK++) //Ecriture des vertex
            {
                pt = myMesh.getVertex(aK);
                fprintf(PlyOut,"%.7f %.7f %.7f\n",pt.x,pt.y,pt.z);
            }
        }

        int width  = aSz.x;
        int height = aSz.y;

        //cout << "myMesh.getFacesNumber()= "<< myMesh.getFacesNumber() << endl;
        for(int i=0 ; i< myMesh.getFacesNumber() ; i++)                          //Ecriture des triangles
        {
            Triangle = myMesh.getTriangle(i);
            Triangle->getVertexesIndexes(t1,t2,t3);              //On recupere les sommets de chaque triangle

            idx = Triangle->getTextureImgIndex();                //Liaison avec l'image correspondante

            //cout << "image pour le triangle " << i << " = " << idx << endl;

            if (idx != valDef)
            {
                Cam = ListCam[idx];

                vector <Pt3dr> Vertex;
                Triangle->getVertexes(Vertex);

                Pt2dr Pt1 = Cam->R3toF2(Vertex[0]);             //projection des sommets du triangle
                Pt2dr Pt2 = Cam->R3toF2(Vertex[1]);
                Pt2dr Pt3 = Cam->R3toF2(Vertex[2]);

                if (Cam->IsInZoneUtile(Pt1) || Cam->IsInZoneUtile(Pt2) || Cam->IsInZoneUtile(Pt3))
                {
                    /*cout << "Pt1= " << Pt1.x << " " << Pt1.y << endl;
                cout << "Pt2= " << Pt2.x << " " << Pt2.y << endl;
                cout << "Pt3= " << Pt3.x << " " << Pt3.y << endl;*/

                    //cout << "idx= " << idx << endl;

                    Pt2dr PtTemp = TabCoor[idx];

                    //cout << "PtTemp = " <<  PtTemp << endl;

                    float Pt1x=(((float)(Pt1.x*Scale)+PtTemp.x)) / (float) width;
                    float Pt1y= 1.0f - (((float)(Pt1.y*Scale)+PtTemp.y)) / (float) height;
                    float Pt2x=(((float)(Pt2.x*Scale)+PtTemp.x)) / (float) width;
                    float Pt2y= 1.0f - (((float)(Pt2.y*Scale)+PtTemp.y)) / (float) height;
                    float Pt3x=(((float)(Pt3.x*Scale)+PtTemp.x)) / (float) width;
                    float Pt3y= 1.0f - (((float)(Pt3.y*Scale)+PtTemp.y)) / (float) height;

                    if (aBin)
                    {
                        WriteType(PlyOut,(unsigned char)3);
                        WriteType(PlyOut,t1);
                        WriteType(PlyOut,t2);
                        WriteType(PlyOut,t3);
                        WriteType(PlyOut,(unsigned char)6);
                        WriteType(PlyOut,Pt1x);
                        WriteType(PlyOut,Pt1y);
                        WriteType(PlyOut,Pt2x);
                        WriteType(PlyOut,Pt2y);
                        WriteType(PlyOut,Pt3x);
                        WriteType(PlyOut,Pt3y);
                    }
                    else
                    {
                        fprintf(PlyOut,"3 %i %i %i ",t1,t2,t3);
                        fprintf(PlyOut,"6 %f %f %f %f %f %f \n",Pt1x,Pt1y,Pt2x,Pt2y,Pt3x,Pt3y);
                    }
                }
                else
                {
                    //cout << "HORS IMG" << endl;
                    if (aBin)
                    {
                        WriteType(PlyOut,(unsigned char)3);
                        WriteType(PlyOut,t1);
                        WriteType(PlyOut,t2);
                        WriteType(PlyOut,t3);
                        WriteType(PlyOut,(unsigned char)6);
                        WriteType(PlyOut,float(0));
                        WriteType(PlyOut,float(0));
                        WriteType(PlyOut,float(0));
                        WriteType(PlyOut,float(0));
                        WriteType(PlyOut,float(0));
                        WriteType(PlyOut,float(0));
                    }
                    else
                    {
                        fprintf(PlyOut,"3 %i %i %i ",t1,t2,t3);
                        fprintf(PlyOut,"6 %d %d %d %d %d %d \n",0,0,0,0,0,0);
                    }
                }
            }
            else
            {
                if (aBin)
                {
                    WriteType(PlyOut,(unsigned char)3);
                    WriteType(PlyOut,t1);
                    WriteType(PlyOut,t2);
                    WriteType(PlyOut,t3);
                    WriteType(PlyOut,(unsigned char)6);
                    WriteType(PlyOut,float(0));
                    WriteType(PlyOut,float(0));
                    WriteType(PlyOut,float(0));
                    WriteType(PlyOut,float(0));
                    WriteType(PlyOut,float(0));
                    WriteType(PlyOut,float(0));
                }
                else
                {
                    fprintf(PlyOut,"3 %i %i %i ",t1,t2,t3);
                    fprintf(PlyOut,"6 %d %d %d %d %d %d \n",0,0,0,0,0,0);
                }
            }
        }
    }
    cout<<"********************************Done*********************************"<<endl;
    cout<<endl;

    return EXIT_SUCCESS;
}

