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

/*#if ELISE_Darwin
    #include <OpenGL/gl.h>
#else
    #include <GL/gl.h>
#endif*/


int Tequila_main(int argc,char ** argv)
{
    bool debug = false;

    int maxTextureSize, texture_units;
    maxTextureSize = texture_units = 0;
    /*glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);

    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &texture_units);

    printf("Max texture size / Max texture units: %d / %d\n", maxTextureSize, texture_units);*/

    if (maxTextureSize == 0) maxTextureSize = 8192;

    std::string aDir,aPat,aFullName,aOri,aPly;
    std::string aOut = "textured_Mesh.ply";
    std::string aTextOut = "UVtexture.jpg";
    int aTextMaxSize = maxTextureSize;
    int aZBuffSSEch = 1;
    int JPGcomp = 70;

    std::stringstream ss  ;
    ss << aTextMaxSize;

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullName,"Full Name (Dir+Pat)",eSAM_IsPatFile)
                            << EAMC(aOri,"Orientation path",eSAM_IsExistDirOri)
                            << EAMC(aPly,"Ply file", eSAM_IsExistFile),
                LArgMain()  << EAM(aOut,"Out",true,"Textured mesh name (def=textured_Mesh.ply)")
                            << EAM(aTextOut,"Texture",true,"Texture name (def=UVtexture.jpg)")
                            << EAM(aTextMaxSize,"Sz",true,"Texture max size (def="+ ss.str() +")")
                            << EAM(aZBuffSSEch,"Scale", true, "Z-buffer downscale factor (def=1)")
                            << EAM(JPGcomp, "QUAL", true, "jpeg compression quality (def=70)")
             );

    if (MMVisualMode) return EXIT_SUCCESS;

    if (aTextMaxSize > maxTextureSize)
    {
        std::stringstream sst  ;
        sst << maxTextureSize;
        cout << "Warning: trying to write texture higher than GL_MAX_TEXTURE_SIZE (" + sst.str() + ")";
        //return;
    }

    SplitDirAndFile(aDir,aPat,aFullName);

    std::string aRes = aDir + StdPrefix(aTextOut) + ".tif";

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    std::list<std::string>  aLS = aICNM->StdGetListOfFile(aPat);

    // If the users enters Ori-MyOrientation/, it will be corrected into MyOrientation
    StdCorrecNameOrient(aOri,aDir);

    std::vector<std::string> ListOri;
    std::vector<CamStenope*> ListCam;

    cout << endl;
    for (std::list<std::string>::const_iterator itS=aLS.begin(); itS!=aLS.end() ; itS++)
    {
        std::string NOri=aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aOri,*itS,true);
        ListOri.push_back(NOri);

        ListCam.push_back(CamOrientGenFromFile(NOri,aICNM));

        cout<<"Image "<<*itS<<", avec son ori : "<< NOri <<endl;
    }

    cout<<endl;
    cout<<"**************************Reading ply file***************************"<<endl;
    cout<<endl;


    cMesh myMesh(aPly);

    int nbVertex = myMesh.getVertexNumber();
    int nbFaces  = myMesh.getFacesNumber();

    //test des normales
    /*Pt3dr barucentr;
    for (int aK=0;aK<nbVertex;aK++)
    {
        barucentr = barucentr + myMesh.getVertex(aK);
    }
    barucentr = barucentr / nbVertex;

    cout << "barycentr= " << barucentr << endl;

    for (int aK=0; aK < nbFaces;aK++)
    {
        cTriangle* Triangle = myMesh.getTriangle(aK);
        Pt3dr nrm = Triangle->getNormale(myMesh, true);

        vector <Pt3dr> Vertex;
        Triangle->getVertexes(myMesh,Vertex);

        Pt3dr centre = (Vertex[0] + Vertex[1] +Vertex[2])/ 3;

        Pt3dr vect = centre - barucentr;
        vect = vect / euclid(vect);

        cout << "norme de vect= " << euclid(vect)<<endl;
        cout << "norme de nrm= " << euclid(nrm)<<endl;

        float scalar = scal(vect, nrm);

        if (scalar < 0.f) cout << "youhou !!!********************************************************" << endl;
    }*/

    printf("Vertex number : %d - faces number : %d \n\n", nbVertex, nbFaces);


    cout<<"*************************Computing Z-Buffer**************************"<< endl;
    cout<< endl;


    vector <cZBuf> aZBuffers;
    vector <Im2D_REAL4> aZBufIm;

    float defValZBuf = 1e9;
    std::list<std::string>::const_iterator itS=aLS.begin();
    for(unsigned int aK=0 ; aK<ListCam.size() ; aK++, itS++) // on teste toutes les CamStenope
    {
        cout << "Z-buffer " << aK+1 << "/" << ListCam.size() << endl;

        cZBuf aZBuffer(ListCam[aK]->Sz(), defValZBuf);

        Im2D_REAL4 res = aZBuffer.BasculerUnMaillage(myMesh, *ListCam[aK]);

        if (debug)
        {
            //conversion du zBuffer en 8 bits
            Pt2di sz = aZBuffer.Sz();
            Im2D_U_INT1 Converted = Im2D_U_INT1(sz.x, sz.y);
            REAL min = FLT_MAX;
            REAL max = 0.f;
            for (int cK=0; cK < sz.x;++cK)
            {
                for (int bK=0; bK < sz.y;++bK)
                {
                    REAL val = res.GetR(Pt2di(cK,bK));

                    if (val != defValZBuf)
                    {
                        if (val > max) max = val;
                        if (val < min) min = val;
                    }
                }
            }

            printf ("Min, max depth = %4.2f %4.2f\n", min, max );

            for (int cK=0; cK < sz.x;++cK)
            {
                for (int bK=0; bK < sz.y;++bK)
                {
                    Converted.SetI(Pt2di(cK,bK),(int)((res.GetR(Pt2di(cK,bK))-min) *255.f/(max-min)));
                }
            }

            std::stringstream ss  ;
            ss << aK;
            string filename = StdPrefix(*itS) + "_zbuf" + ss.str() + ".tif";
            printf ("Saving %s\n", filename.c_str());
            Tiff_Im::CreateFromIm(Converted, filename);
            printf ("Done\n");

            //image des labels
            Im2D_INT4 Labels = aZBuffer.getIndexImage();

            filename = StdPrefix(*itS) + "_label" + ss.str() + ".tif";
            printf ("Saving %s\n", filename.c_str());
            Tiff_Im::CreateFromIm(Labels, filename);
            printf ("Done\n");
        }

        aZBufIm.push_back(res);

        aZBuffers.push_back(aZBuffer);
    }

    cout << endl;
    cout << "************************Choosing best image**************************" << endl;
    cout << endl;

    vector<int> vIndex; //vIndex[aK] indique quelle image est vue par le triangle aK
    int valDef = -1;

    for(int i=0 ; i < nbFaces; i++)                            //Pour un triangle
    {
        cTriangle * Triangle = myMesh.getTriangle(i);
        Pt3dr Norm = Triangle->getNormale(true);       //Et sa normale

        float PScalcur = -.4f; //angle min = cos(120) = -0.5
        int index = valDef;

        for(unsigned int j=0 ; j<ListCam.size() ; j++) // on teste toutes les CamStenope
        {
            vector <unsigned int> vTri = aZBuffers[j].getVisibleTrianglesIndexes();

            if (std::find(vTri.begin(),vTri.end(), i) != vTri.end())
            {

                double PScalnew = scal(Norm, ListCam[j]->DirK());
                //cout << "scal= " << PScalnew << endl;
                if(PScalnew<PScalcur)        //On garde celle pour laquelle le PS est le plus fort
                {
                    PScalcur = PScalnew;
                    index = j;

                    Triangle->setTextured(true);
                }
            }
        }

        vIndex.push_back(index);
    }

    std::vector <int> index;

    for(unsigned int aK=0; aK <vIndex.size();++aK)
    {
        int curIndex = vIndex[aK];

        if (std::find(index.begin(), index.end(), curIndex)==index.end() && curIndex != valDef)
        {
            index.push_back(curIndex);
        }
    }

    cout << "selected images / total : " << index.size() << " / " << aLS.size() << endl;

    cout << endl;
    cout <<"*******************Filtering border triangles**********************"<<endl;
    cout << endl;

    int maxIter = 2;
    int iter = 0;
    bool cond = true;

    while (cond && iter < maxIter)
    {
        cout << "myMesh.getFacesNb " << myMesh.getFacesNumber() << endl;
        vector<int> vRemovedTri = myMesh.clean();

        cout << "myMesh.getFacesNb " << myMesh.getFacesNumber() << endl;

        for (unsigned int aK=0; aK < vRemovedTri.size(); ++aK)
            vIndex.erase(vIndex.begin()+vRemovedTri[aK]);

        iter++;
        cout << "round " << iter << endl;

        cond = false;
        for (int aK=0; aK< myMesh.getFacesNumber();++aK)
            if (myMesh.getTriangle(aK)->getEdgesNumber() < 3 )
            {
                cond =true;
                break;
            }
    }


    cout << "Faces number after filtering: " << myMesh.getFacesNumber() << endl;

    cout << endl;
    cout <<"************************Writing texture************************"<<endl;
    cout << endl;

    Pt2di aSzMax;
    std::vector<Tiff_Im> aVT;     //Vecteur contenant les images
    std::vector<Pt2dr> TabCoor;   //Vecteur contenant les coordonnées des images dans la texture
    int aNbCh = 0;

    vector <Im2D_REAL4> final_ZBufIm;

    std::sort(index.begin(), index.end());

    for (unsigned int aK=0; aK < index.size() ; aK++)
    {
        int id = index[aK];
        int bK=0;
        for (std::list<std::string>::const_iterator itS=aLS.begin(); itS!=aLS.end() ; itS++, bK++)
        {
            if (id == bK)
            {
                aVT.push_back(Tiff_Im::StdConvGen(aDir+*itS,-1,false,true));
                final_ZBufIm.push_back(aZBufIm[id]);
                aSzMax.SetSup(aVT.back().sz());
                aNbCh = ElMax(aNbCh,aVT.back().nb_chan());
                break;
            }
        }
    }

    int aNbLine = round_up(sqrt(double(aVT.size())));
    int aNbCol = round_up(aVT.size()/double(aNbLine));

    cout<< "Texture de " << aNbLine << " lignes et "  << aNbCol <<" colonnes, contenant "<< aVT.size() <<" images. "<< endl;
    cout<<endl;

    int full_width  = aSzMax.x * aNbCol;
    int full_height = aSzMax.y * aNbLine;

    float Scale = (float) aTextMaxSize / ElMax(full_width, full_height) ;

    cout << "Scaling factor = " << Scale << endl;

    int final_width  = round_up(full_width * Scale);
    int final_height = round_up(full_height * Scale);

    Pt2di aSz (
                final_width,       //A modifier pour le scale
                final_height      //A modifier pour le scale
                );

    cout << "SZ = " << aSz << " :: " << aNbCol << " X " << aNbLine  << "\n";

    Tiff_Im::PH_INTER_TYPE aPhI = aVT[0].phot_interp();
    if (aNbCh==3)
        aPhI = Tiff_Im::RGB;

    Tiff_Im  FileRes
            (
                aRes.c_str(),
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

    std::string newName = StdPrefix(aRes) + ".jpg ";
    std::stringstream st  ;
    st << JPGcomp;

    std::string aCom =  g_externalToolHandler.get( "convert" ).callName() + std::string(" -quality ") + st.str() + " "
            + aRes + " " + newName;

    cout << "COM= " << aCom << endl;

    system_call(aCom.c_str());

    aCom = std::string(SYS_RM) + " " + aRes;
    system_call(aCom.c_str());

    cout << endl;
    cout <<"************************Writing ply file*************************"<<endl;
    cout <<endl;


    FILE * PlyOut = FopenNN(aOut,"w", "UV Mapping");         //Ecriture du header
    fprintf(PlyOut,"ply\n");
    fprintf(PlyOut,"format ascii 1.0\n");
    fprintf(PlyOut,"comment UV Mapping generated\n");
    fprintf(PlyOut,"comment TextureFile %s\n", newName.c_str());
    fprintf(PlyOut,"element vertex %i\n",nbVertex);
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

    for(int aK=0 ; aK<nbVertex ; aK++) //Ecriture des vertex
    {
        pt = myMesh.getVertex(aK);
        fprintf(PlyOut,"%.7f %.7f %.7f\n",pt.x,pt.y,pt.z);
    }

    int width  = aSz.x;
    int height = aSz.y;

    cout << "vIndex.size= "<< vIndex.size() << endl;
    cout << "myMesh.getFacesNumber()= "<< myMesh.getFacesNumber() << endl;
    for(int i=0 ; i< myMesh.getFacesNumber() ; i++)                          //Ecriture des triangles
    {
        Triangle = myMesh.getTriangle(i);
        Triangle->getVertexesIndexes(t1,t2,t3);              //On recupere les sommets de chaque triangle

        idx=vIndex[i];                                      //Liaison avec l'image correspondante

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

                Pt2dr PtTemp = TabCoor[idx];

                //cout << "PtTemp = " <<  PtTemp << endl;

                float Pt1x=(((float)(Pt1.x*Scale)+PtTemp.x)) / (float) width;
                float Pt1y= 1.0f - (((float)(Pt1.y*Scale)+PtTemp.y)) / (float) height;
                float Pt2x=(((float)(Pt2.x*Scale)+PtTemp.x)) / (float) width;
                float Pt2y= 1.0f - (((float)(Pt2.y*Scale)+PtTemp.y)) / (float) height;
                float Pt3x=(((float)(Pt3.x*Scale)+PtTemp.x)) / (float) width;
                float Pt3y= 1.0f - (((float)(Pt3.y*Scale)+PtTemp.y)) / (float) height;


                fprintf(PlyOut,"3 %i %i %i ",t1,t2,t3);
                fprintf(PlyOut,"6 %f %f %f %f %f %f \n",Pt1x,Pt1y,Pt2x,Pt2y,Pt3x,Pt3y);
            }
            else
            {
                //cout << "HORS IMG" << endl;
                fprintf(PlyOut,"3 %i %i %i ",t1,t2,t3);
                fprintf(PlyOut,"6 %f %f %f %f %f %f \n",0.0,0.0,0.0,0.0,0.0,0.0);
            }


        }
        else
        {
            fprintf(PlyOut,"3 %i %i %i ",t1,t2,t3);
            fprintf(PlyOut,"6 %f %f %f %f %f %f \n",0.0,0.0,0.0,0.0,0.0,0.0);
        }
    }

    cout<<"*******************************Done*********************************"<<endl;
    cout<<endl;

    return EXIT_SUCCESS;
}
