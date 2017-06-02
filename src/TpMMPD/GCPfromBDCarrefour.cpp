#include "Sat.h"
#include "../uti_phgrm/MICMAC/cCameraModuleOrientation.h"
#include "../uti_phgrm/MICMAC/cInterfModuleImageLoader.h"
#include "GCPfromBDAmer.h"

#ifdef WIN32
#else
#ifndef __APPLE__
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

//pour la connexion au GPP
std::string aKeyGPP;
std::string aHttpProxy;

///
std::string getCurrentDir()
{
    char *path=NULL;
    size_t size = 0;
    path=getcwd(path,size);
    return std::string(path);
}

///
///
///
bool getPointsFile(const std::string aGCPFileName, std::map<std::string,Pt3dr>& mGCP)
{
    std::ifstream fic (aGCPFileName.c_str());
    if (!fic.good())
    {
        cerr << "Error in reading "<<aGCPFileName<< " !" << std::endl;
        return false;
    }
    
    std::string line;
    while ( std::getline(fic, line ) )
    {
        Pt3dr Pt; std::string nPt;
        std::istringstream iss4(line);
        iss4 >> nPt >> Pt.x >> Pt.y >> Pt.z;
        mGCP.insert(std::pair<std::string,Pt3dr>(nPt,Pt));
    }
    
    fic.close();
    return true;
}


///
///
///
bool getPoints(const std::string GCPFiles, std::map<std::string,Pt3dr>& mGCP, const std::string PatCtrlPt)
{
    std::list<std::string> aLFile = GCPfromBDI_files::getFiles(GCPFiles);
    
    std::list<std::string>::const_iterator it (aLFile.begin()),fin=aLFile.end();
    for(;it!=fin;++it)
        getPointsFile(*it,mGCP);
    
    return true;
}

///
///
///
std::string getGPPAccess()
{
    std::ostringstream oss;
    oss << g_externalToolHandler.get( "curl" ).callName() << " -H='Referer: http://localhost' ";
    if (!aHttpProxy.empty())
        oss << "-x "<<aHttpProxy;
    return oss.str();
}

///
///
///
bool getVignetteGPP(const std::string nomDalleOrtho, const Pt3dr& ptSol,
                    const Pt2di SzOrtho, const float resol)
{
    std::string gppAccess = getGPPAccess();
    
    //dalle
    double xminDalle = ptSol.x - SzOrtho.x*resol/2;
    double yminDalle = ptSol.y - SzOrtho.y*resol/2;
    double xmaxDalle = xminDalle + SzOrtho.x*resol;
    double ymaxDalle = yminDalle + SzOrtho.y*resol;
    
    //recuperation de l'ortho
    std::ostringstream ossGPP;
    ossGPP << std::fixed << gppAccess << " -o "<<nomDalleOrtho<< " \"http://wxs-i.ign.fr/"<<aKeyGPP<<"/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&STYLES=normal&FORMAT=image/geotiff&BBOX="<< xminDalle<<","<<yminDalle<<","<<xmaxDalle<<","<<ymaxDalle<<"&CRS=EPSG:2154&WIDTH="<<SzOrtho.x<<"&HEIGHT="<<SzOrtho.y<<"\"";
    
    std::cout << ossGPP.str() << std::endl;
    
    system(ossGPP.str().c_str());
    
    std::string vignetteName = getCurrentDir() + "/" + nomDalleOrtho;
    return (ELISE_fp::exist_file(vignetteName));
}

/*
 ///
 ///
 ///
 bool getVignetteGPP(const std::string nomDalleOrtho, const Pt3dr& ptSol,
 const Pt2di SzOrtho, const float resol)
 {
 std::string gppAccess = getGPPAccess();
 
 //dalle
 double xminDalle = ptSol.x - SzOrtho.x*resol/2;
 double yminDalle = ptSol.y - SzOrtho.y*resol/2;
 double xmaxDalle = xminDalle + SzOrtho.x*resol;
 double ymaxDalle = yminDalle + SzOrtho.y*resol;
 
 double resolutionOrtho=0.20; // connu par defaut...
 //taille de l'ortho a pleine résolution :
 int Nc = (xmaxDalle-xminDalle)/resolutionOrtho+1;
 int Nl = (ymaxDalle-yminDalle)/resolutionOrtho+1;
 
 //recuperation de l'ortho
 std::ostringstream ossGPP;
 ossGPP << std::fixed << gppAccess << " -o "<<nomDalleOrtho<< " \"http://wxs-i.ign.fr/"<<aKeyGPP<<"/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&STYLES=normal&FORMAT=image/geotiff&BBOX="<< xminDalle<<","<<yminDalle<<","<<xmaxDalle<<","<<ymaxDalle<<"&CRS=EPSG:2154&WIDTH="<<Nc<<"&HEIGHT="<<Nl<<"\"";
 
 std::cout << ossGPP.str() << std::endl;
 
 system(ossGPP.str().c_str());
 
 std::string vignetteName = getCurrentDir() + "/" + nomDalleOrtho;
 return (ELISE_fp::exist_file(vignetteName));
 }
 */

///
///
///
bool processPtWithIm(const std_unique_ptr<ElCamera>& aCam,
                     const std::string aNameImage,
                     const std::string nomPoint,
                     const Pt3dr& Pt,
                     const Pt2di& SzVignette,
                     const double CorefMinAccept,
                     GCPfromBDI_Export::MPtCoordPix& mPix)
{
    bool verbose  (1);
    bool verbose2 (0);
    
    double  NoData = -1.;
    int     marge = 10;
    
    if (!aCam->PIsVisibleInImage(Pt)) return false;
    
    Pt2dr ptImage = aCam->Ter2Capteur(Pt);
    float resolImageSat = floor(100*aCam->ResolutionSol(Pt))/100;
    
    std::cout << std::fixed << std::setprecision(5) << std::endl;
    if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] point " << nomPoint << " : " << Pt.x << " " << Pt.y << std::endl;
    if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] point in input image " << ptImage.x << " " << ptImage.y << std::endl;
    if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] input image resolution : "<< resolImageSat << std::endl;
    
    //calcul d'une pseudoOrtho :
    Pt2di POSz (SzVignette.x+2*marge,SzVignette.y+2*marge);
    Pt3dr PNOpseudoOrtho (Pt.x-POSz.x*resolImageSat/2,Pt.y+POSz.y*resolImageSat/2,Pt.z);
    std_unique_ptr<TIm2D<U_INT2,INT4> > pseudoOrtho (new TIm2D<U_INT2,INT4>(POSz));
    if (!GCPfromBDI_Image::calcPseudoOrtho(aNameImage,PNOpseudoOrtho,POSz,resolImageSat,aCam,pseudoOrtho))
        return false;
    
    //recuperation de la vignette
    std::ostringstream ossNomOrtho;
    ossNomOrtho << "Ortho_"<<nomPoint<<".tif";
    std::string nomDalleOrtho = ossNomOrtho.str();
    if (!getVignetteGPP(nomDalleOrtho,Pt,SzVignette,resolImageSat))
        return false; // un pb dans la lecture de la BDOrtho
    
    //BDOrtho => tab
    std::vector<double> fenOrtho;
    std_unique_ptr<TIm2D<U_INT2,INT4> > cropBDOrtho (createTIm2DFromFile<U_INT2,INT4>(nomDalleOrtho,Pt2di(0,0),SzVignette));
    GCPfromBDI_Correl::getVecFromTIm2D(cropBDOrtho,fenOrtho,NoData);
    
    //export de la pseudoOrtho
    std::ostringstream ossNomPseudoOrtho;
    ossNomPseudoOrtho << "PseudoOrtho_"<<nomPoint<<".tif";
    std::string nomPseudoOrtho = ossNomPseudoOrtho.str();
    Tiff_Im out(nomPseudoOrtho.c_str(), pseudoOrtho->sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
    ELISE_COPY(pseudoOrtho->_the_im.all_pts(),pseudoOrtho->_the_im.in(),out.out());
    
    //correlations :
    int cc(0),ll(0);
    double CoefCorelMax(-1);
    GCPfromBDI_Export::ItPtCoordPix itPix;
    for (size_t l(0); l<POSz.y-SzVignette.y;l++)
    {
        for (size_t c(0); c<POSz.x-SzVignette.x;c++)
        {
            double cofCorel =  GCPfromBDI_Correl::getCorelCoef(Pt2di(c,l),SzVignette,nomPseudoOrtho,fenOrtho,NoData);
            
            if(cofCorel>=CoefCorelMax)
            {
                if (cofCorel<CorefMinAccept) // pas assez bon
                    continue;
                
                Pt2di ptCentralPO (c+SzVignette.x/2,l+SzVignette.y/2);
                Pt3dr ptCentralPOTer ( PNOpseudoOrtho.x + ptCentralPO.x*resolImageSat,
                                      PNOpseudoOrtho.y - ptCentralPO.y*resolImageSat,
                                      PNOpseudoOrtho.z);
                Pt2dr ptImage_i = aCam->Ter2Capteur(ptCentralPOTer);
                
                CoefCorelMax = cofCorel;
                cc=c; ll=l;
                
                itPix = mPix.find(nomPoint);
                Pt2dr ptr (ptImage_i.x,ptImage_i.y);
                if (itPix==mPix.end())
                    mPix.insert(std::pair<std::string,Pt2dr>(nomPoint,ptr));
                else
                    itPix->second = ptr;
            }
        }
    }
    
    //on purge les images créés :
    ELISE_fp::RmFile(getCurrentDir()+"/"+nomDalleOrtho.c_str());
    ELISE_fp::RmFile(getCurrentDir()+"/"+nomPseudoOrtho.c_str());
    std::string nomOrthoEnPlus = getCurrentDir()+"/Tmp-MM-Dir/" + nomDalleOrtho.c_str() + "_Ch1.tif";
    ELISE_fp::RmFile(nomOrthoEnPlus);
    
    itPix = mPix.find(nomPoint);
    if (itPix == mPix.end()) return false; // on a pas trouve de coef de correlation suffisant...
    
    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] " << CoefCorelMax << " : (" << ptImage.x << ";" << ptImage.y << ") => (" << itPix->second.x << ";" << itPix->second.y << ")" << std::endl;
    
    return true;
}

///
///
///
int GCP_From_BDCarrefour_main(int argc, char **argv)
{
    bool verbose  (1);
    
    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] Get BDOrtho control point for bundle" << std::endl;
    
    // EAMC
    std::string GCPFiles;
    std::string aOutDir;
    std::string ImageNames;
    std::string PatCtrlPt;
    
    // EAM
    std::string aGRIDExt("GRI");
    int sizeCorrelation(200);
    double CorefMinAccept(0.5);
    
    ElInitArgMain
    (
     argc, argv,
     LArgMain() << EAMC(GCPFiles,"Full Name (Dir+Pat)")
     << EAMC(ImageNames,"Full Name (Dir+Pat)")
     << EAMC(aOutDir,"Name of output directory")
     << EAMC(aKeyGPP,"GPP Key"),
     LArgMain() << EAM(aHttpProxy,"Proxy",true,"http proxy for GPP access")
     << EAM(sizeCorrelation,"CorSw",200,"Size Correlation Windows")
     << EAM(aGRIDExt,"Grid","GRI","Orientation file extension")
     << EAM(CorefMinAccept,"CoefMini",0.5,"Minimum Coefficient to be acceptated")
     );
    
    //lecture des points de contrôle
    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] Lecture des points de controles "<< std::endl;
    GCPfromBDI_Export::MPtCoordTer mGCP;
    if (!getPoints(GCPFiles,mGCP,PatCtrlPt)) return EXIT_FAILURE;
    
    
    Pt2di SzVignette (sizeCorrelation,sizeCorrelation);
    
    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] taille de la vignette : " << SzVignette.x << " " << SzVignette.y << std::endl;
    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] Nb de points          : " << mGCP.size() << std::endl;
    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] orientation de type   : " << aGRIDExt << std::endl;
    
    //ecriture du fichier de point terrain :
    std::ostringstream ossNomPt3D;
    ossNomPt3D << aOutDir << "/gcp-3D.xml";
    GCPfromBDI_Export::WriteGCP3D(mGCP,ossNomPt3D.str());
    
    GCPfromBDI_Export::MPtImage mPtImage;
    
    std::list<std::string> lImage = GCPfromBDI_files::getFiles(ImageNames);
    std::list<std::string>::const_iterator itImage(lImage.begin()),finImage(lImage.end());
    for (;itImage!=finImage;++itImage)
    {
        
        std::string aNameImageOri;
        if (!GCPfromBDI_Image::getOri(*itImage,aGRIDExt,aNameImageOri))
        {
            std::cout << "No orientation file for " << *itImage << " - skip " << std::endl;
            continue;
        }
        
        // chargement de l'orientation de l'image sat
        ElAffin2D oriIntImaM2C;
        Pt2di ImgSz = GCPfromBDI_Image::ImageSize(*itImage);
        std_unique_ptr<ElCamera> aCam (new cCameraModuleOrientation(new OrientationGrille(aNameImageOri),ImgSz,oriIntImaM2C));
        
        if (verbose) std::cout << "[GCP_From_BDCarrefour_main] for each points..." << std::endl;
        std::map<std::string,Pt3dr>::iterator itGCP (mGCP.begin()), endGCP (mGCP.end());
        GCPfromBDI_Export::MPtCoordPix mPix; GCPfromBDI_Export::cItPtCoordPix itPix; size_t i(1);
        for (;itGCP!=endGCP;itGCP++)
        {
            if (verbose) std::cout << "[GCP_From_BDCarrefour_main] Pt " << i <<"/"<<mGCP.size() << std::endl; i++;
            std::string nomPoint = itGCP->first;
            Pt3dr& Pt = itGCP->second;
            processPtWithIm(aCam, *itImage, nomPoint, Pt, SzVignette, CorefMinAccept, mPix);
        }
        
        mPtImage.insert(GCPfromBDI_Export::pPtImage(*itImage,mPix));
    }
    
    std::ostringstream ossNomPt2D;
    ossNomPt2D << aOutDir << "/gcp-2D.xml";
    GCPfromBDI_Export::WriteGCP2D(mPtImage,ossNomPt2D.str());
    
    //on purge les images créés :
    //std::remove("Tmp-MM-Dir");
    
    return 0;
}

#ifdef WIN32
#else
#ifndef __APPLE__
#pragma GCC diagnostic pop
#endif
#endif


