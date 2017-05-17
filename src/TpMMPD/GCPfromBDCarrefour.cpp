#include "Sat.h"
#include "../uti_phgrm/MICMAC/cCameraModuleOrientation.h"
#include "../uti_phgrm/MICMAC/Jp2ImageLoader.h"
#include "../uti_phgrm/MICMAC/cInterfModuleImageLoader.h"

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

//
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
double correlVec(std::vector<double> const &f1,std::vector<double> const &f2)
{
    std::vector<double>::const_iterator it1,it2,fin1,fin2;
    it1=f1.begin();
    it2=f2.begin();
    fin1=f1.end();
    fin2=f2.end();
    
    double sa=0.,sa2=0.,sb=0.,sb2=0,sab=0.;
    int N=0;
    for(;(it1!=fin1)&&(it2!=fin2);)
    {
        sa += (*it1);
        sa2 += pow((*it1),2);
        sb += (*it2);
        sb2 += pow((*it2),2);
        sab += (*it1)*(*it2);
        ++N;
        ++it1;
        ++it2;
    }
    double pdtVar = (sa2-pow(sa,2)/N)*(sb2-pow(sb,2)/N);
    if (pdtVar!=0.)
        return (sab-sa*sb/N)/sqrt(pdtVar);
    return 0.;
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

///
///
///
void getVecFromTIm2D(const std_unique_ptr<TIm2D<U_INT2,INT4> >& cropImg, std::vector<double>& fenImage,
                     const double NoData)
{
    for (size_t l(0); l<cropImg->sz().y; l++)
    {
        for (size_t c(0); c<cropImg->sz().x; c++)
        {
            fenImage.push_back(cropImg->getr(Pt2dr(c,l),NoData));
        }
    }
}

///
///
///
double getCorelCoef(const Pt2di& ptImage, const Pt2di& orthoSz, const std::string nomImage,
                    const std::vector<double> fenRef, const double NoData)
{
    std::vector<double> fenImage_i;
    std_unique_ptr<TIm2D<U_INT2,INT4> > cropImg (createTIm2DFromFile<U_INT2,INT4>(nomImage,ptImage,orthoSz));
    getVecFromTIm2D(cropImg,fenImage_i,NoData);
    
    //correlation :
    return correlVec(fenRef,fenImage_i);
}

///
///
///
Pt2di ImageSize(std::string const &aName)
{
    //on recupere l'extension
    int placePoint = -1;
    for(int l=(int)(aName.size()-1);(l>=0)&&(placePoint==-1);--l)
    {
        if (aName[l]=='.')
        {
            placePoint = l;
        }
    }
    std::string ext = std::string("");
    if (placePoint!=-1)
    {
        ext.assign(aName.begin()+placePoint+1,aName.end());
    }
    
#if defined (__USE_JP2__)
    // on teste l'extension
    if ((ext==std::string("jp2"))|| (ext==std::string("JP2")) || (ext==std::string("Jp2")))
    {
        //std::cout<<"JP2 avec Jp2ImageLoader"<<std::endl;
        std_unique_ptr<cInterfModuleImageLoader> aRes(new JP2ImageLoader(aName, false));
        if (aRes.get())
        {
            return Std2Elise(aRes->Sz(1));
        }
    }
#endif
    
    Tiff_Im aTif = Tiff_Im::StdConvGen(aName,1,true,false);
    return aTif.sz();
}

///
///
///
double minVec(std::vector<double>& c)
{
    double ret = c[0];
    for (size_t i(1); i<c.size(); i++)
        if (c[i]<ret)
            ret = c[i];
    return ret;
}
double maxVec(std::vector<double>& c)
{
    double ret = c[0];
    for (size_t i(1); i<c.size(); i++)
        if (c[i]>ret)
            ret = c[i];
    return ret;
}

///
///
///
bool calcPseudoOrtho(const std::string& aNameImage, const Pt3dr& PtNO, const Pt2di& size,
                     const float resol, const std_unique_ptr<ElCamera>& aCam,
                     std_unique_ptr<TIm2D<U_INT2,INT4> >& vignette)
{
    
    Pt3dr PtNE (PtNO.x+size.x*resol,    PtNO.y,                 PtNO.z);
    Pt3dr PtSO (PtNO.x,                 PtNO.y-size.y*resol,    PtNO.z);
    Pt3dr PtSE (PtNO.x+size.x*resol,    PtNO.y-size.y*resol,    PtNO.z);

    Pt2dr ptINO = aCam->Ter2Capteur(PtNO);
    Pt2dr ptINE = aCam->Ter2Capteur(PtNE);
    Pt2dr ptISO = aCam->Ter2Capteur(PtSO);
    Pt2dr ptISE = aCam->Ter2Capteur(PtSE);
    
    std::vector<double> vc; vc.push_back(ptINO.x); vc.push_back(ptINE.x); vc.push_back(ptISO.x); vc.push_back(ptISE.x);
    std::vector<double> vl; vl.push_back(ptINO.y); vl.push_back(ptINE.y); vl.push_back(ptISO.y); vl.push_back(ptISE.y);

    int cmin = floor(minVec(vc));
    int cmax = floor(maxVec(vc))+1;
    int lmin = floor(minVec(vl));
    int lmax = floor(maxVec(vl))+1;

    int buffer (1);
    Pt2di minIm  (cmin-buffer,lmin-buffer);
    if (minIm.x<0 || minIm.y<0) return false;
    
    Pt2di sizeImSrc = ImageSize(aNameImage);
    Pt2di sizeIm (cmax-cmin+2*buffer,lmax-lmin+2*buffer);
    
    if ( minIm.x + sizeIm.x >= sizeImSrc.x || minIm.y + sizeIm.y >=sizeImSrc.y ) return false;
    
    std_unique_ptr<TIm2D<U_INT2,INT4> > cropImg_cl (createTIm2DFromFile<U_INT2,INT4>(aNameImage,minIm,sizeIm));
    
    for (size_t l(0); l<size.y ; l++)
    {
        for (size_t c(0); c<size.x ; c++)
        {
            Pt3dr pt (PtNO.x + c*resol, PtNO.y - l*resol, PtNO.z);
            Pt2dr ptImage = aCam->Ter2Capteur(pt);
            Pt2dr ptImageCrop (ptImage.x-minIm.x,ptImage.y-minIm.y);

            double radio = cropImg_cl->getr(ptImageCrop);
            vignette->oset(Pt2di(c,l),(int)radio);
        }
    }
    
    return true;
}

///
///
///
void WriteGCP3D(const std::map<std::string,Pt3dr>& mGCP, const std::string nomFic)
{
    std::ofstream fic (nomFic.c_str());
    fic << std::fixed << std::setprecision(5);
    fic << "<?xml version=\"1.0\" ?>" << std::endl;
    fic << "<DicoAppuisFlottant>" << std::endl;
    
    std::map<std::string,Pt3dr>::const_iterator itGCP (mGCP.begin()), endGCP (mGCP.end());
    for (;itGCP!=endGCP;itGCP++)
    {
        fic << "    <OneAppuisDAF>" << std::endl;
        fic << "        <Pt>" << itGCP->second.x << " " << itGCP->second.y << " " << itGCP->second.z << "</Pt>" << std::endl;
        fic << "        <NamePt>" << itGCP->first << "</NamePt>" << std::endl;
        fic << "        <Incertitude>" << "1 1 1" << "</Incertitude>" << std::endl;//pour le moment on ne joue pas avec les incertitudes
        fic << "    </OneAppuisDAF>" << std::endl;
    }

    fic << "</DicoAppuisFlottant>" << std::endl;
    fic.close();
}

typedef std::map<std::string,Pt2dr>                             MPtCoordPix;
typedef std::map<std::string,Pt2dr>::iterator                   ItPtCoordPix;
typedef std::map<std::string,Pt2dr>::const_iterator             cItPtCoordPix;
typedef std::map<std::string,MPtCoordPix>                       MPtImage;
typedef std::map<std::string,MPtCoordPix>::const_iterator       ItPtImage;
typedef std::pair<std::string,MPtCoordPix>                      pPtImage;

///
///
///
void WriteGCP2D(const MPtImage& mptImage, const std::string nomFic)
{
    std::ofstream fic (nomFic.c_str());
    fic << std::fixed << std::setprecision(5);
    fic << "<?xml version=\"1.0\" ?>" << std::endl;
    fic << "<SetOfMesureAppuisFlottants>" << std::endl;

    ItPtImage itImage (mptImage.begin()), endImage (mptImage.end());
    for (;itImage!=endImage;itImage++)
    {
        fic << "    <MesureAppuiFlottant1Im>"<<std::endl;
        fic << "        <NameIm>" << itImage->first <<"</NameIm>"<<std::endl;

        const MPtCoordPix& mptCoord = itImage->second;
        cItPtCoordPix itPt (mptCoord.begin()), endPt (mptCoord.end());
        for (;itPt!=endPt;itPt++)
        {
            fic << "        <OneMesureAF1I>"<<std::endl;
            fic << "            <NamePt>" << itPt->first << "</NamePt>"<<std::endl;
            fic << "            <PtIm>" << itPt->second.x << " " << itPt->second.y << "</PtIm>"<<std::endl;
            fic << "        </OneMesureAF1I>"<<std::endl;
        }
        
        fic << "    </MesureAppuiFlottant1Im>"<<std::endl;
    }
    
    fic << "</SetOfMesureAppuisFlottants>" << std::endl;
    fic.close();
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
std::list<std::string> getFiles(const std::string fileNameWithPattern)
{
    std::string aDir,aPat;
    SplitDirAndFile(aDir,aPat,fileNameWithPattern);
    cInterfChantierNameManipulateur *aICNM;
    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    std::list<std::string> listFinal, list = aICNM->StdGetListOfFile(aPat);
    
    std::list<std::string>::const_iterator it (list.begin()),fin=list.end();
    for(;it!=fin;++it)
        listFinal.push_back(aDir+"/"+*it);
    
    return listFinal;
}

///
///
///
bool getPoints(const std::string GCPFiles, std::map<std::string,Pt3dr>& mGCP, const std::string PatCtrlPt)
{
    std::list<std::string> aLFile = getFiles(GCPFiles);

    std::list<std::string>::const_iterator it (aLFile.begin()),fin=aLFile.end();
    for(;it!=fin;++it)
        getPointsFile(*it,mGCP);
    
    return true;
}

///
///
///
bool processPtWithIm(const std_unique_ptr<ElCamera>& aCam,
                     const std::string aNameImage,
                     const std::string nomPoint,
                     const Pt3dr& Pt,
                     const Pt2di& SzVignette,
                     const double CorefMinAccept,
                     MPtCoordPix& mPix)
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
    if (!calcPseudoOrtho(aNameImage,PNOpseudoOrtho,POSz,resolImageSat,aCam,pseudoOrtho))
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
    getVecFromTIm2D(cropBDOrtho,fenOrtho,NoData);
    
    //export de la pseudoOrtho
    std::ostringstream ossNomPseudoOrtho;
    ossNomPseudoOrtho << "PseudoOrtho_"<<nomPoint<<".tif";
    std::string nomPseudoOrtho = ossNomPseudoOrtho.str();
    Tiff_Im out(nomPseudoOrtho.c_str(), pseudoOrtho->sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
    ELISE_COPY(pseudoOrtho->_the_im.all_pts(),pseudoOrtho->_the_im.in(),out.out());
    
    //correlations :
    int cc(0),ll(0);
    double CoefCorelMax(-1);
    ItPtCoordPix itPix;
    for (size_t l(0); l<POSz.y-SzVignette.y;l++)
    {
        for (size_t c(0); c<POSz.x-SzVignette.x;c++)
        {
            double cofCorel =  getCorelCoef(Pt2di(c,l),SzVignette,nomPseudoOrtho,fenOrtho,NoData);
            
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
int getPlacePoint(const std::string fileName)
{
    int placePoint = -1;
    for(int l=(int)(fileName.size()-1);(l>=0)&&(placePoint==-1);--l)
    {
        if (fileName[l]=='.')
        {
            placePoint = l;
        }
    }
    return placePoint;
}

///
///
///
bool getOri(const std::string imageName, const std::string aGRIDExt, std::string& orientationName)
{
    int placePoint = getPlacePoint(imageName);
    if (placePoint==-1) return false;
    
    std::string baseName;
    baseName.assign(imageName.begin(),imageName.begin()+placePoint+1);
    
    orientationName = baseName+aGRIDExt;
    return (ELISE_fp::exist_file(orientationName));
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
    std::map<std::string,Pt3dr> mGCP;
    if (!getPoints(GCPFiles,mGCP,PatCtrlPt)) return EXIT_FAILURE;
    
    Pt2di SzVignette (sizeCorrelation,sizeCorrelation);
    
    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] taille de la vignette : " << SzVignette.x << " " << SzVignette.y << std::endl;
    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] Nb de points          : " << mGCP.size() << std::endl;
    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] orientation de type   : " << aGRIDExt << std::endl;
    
    //ecriture du fichier de point terrain :
    std::ostringstream ossNomPt3D;
    ossNomPt3D << aOutDir << "/gcp-3D.xml";
    WriteGCP3D(mGCP,ossNomPt3D.str());
    
    MPtImage mPtImage;
    
    std::list<std::string> lImage = getFiles(ImageNames);
    std::list<std::string>::const_iterator itImage(lImage.begin()),finImage(lImage.end());
    for (;itImage!=finImage;++itImage)
    {

        std::string aNameImageOri;
        if (!getOri(*itImage,aGRIDExt,aNameImageOri))
        {
            std::cout << "No orientation file for " << *itImage << " - skip " << std::endl;
            continue;
        }
        
        // chargement de l'orientation de l'image sat
        ElAffin2D oriIntImaM2C;
        Pt2di ImgSz = ImageSize(*itImage);
        std_unique_ptr<ElCamera> aCam (new cCameraModuleOrientation(new OrientationGrille(aNameImageOri),ImgSz,oriIntImaM2C));
        
        if (verbose) std::cout << "[GCP_From_BDCarrefour_main] for each points..." << std::endl;
        std::map<std::string,Pt3dr>::iterator itGCP (mGCP.begin()), endGCP (mGCP.end());
        MPtCoordPix mPix; cItPtCoordPix itPix; size_t i(1);
        for (;itGCP!=endGCP;itGCP++)
        {
            if (verbose) std::cout << "[GCP_From_BDCarrefour_main] Pt " << i <<"/"<<mGCP.size() << std::endl; i++;
            std::string nomPoint = itGCP->first;
            Pt3dr& Pt = itGCP->second;
            processPtWithIm(aCam, *itImage, nomPoint, Pt, SzVignette, CorefMinAccept, mPix);
        }
        
        mPtImage.insert(pPtImage(*itImage,mPix));
    }
    
    std::ostringstream ossNomPt2D;
    ossNomPt2D << aOutDir << "/gcp-2D.xml";
    WriteGCP2D(mPtImage,ossNomPt2D.str());
    
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


