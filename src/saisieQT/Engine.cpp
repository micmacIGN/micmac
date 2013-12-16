#include "Engine.h"
#include "general/bitm.h"

cLoader::cLoader()
 : _FilenamesIn(),
   _FilenamesOut(),
   _postFix("_Masq")
{}

void cLoader::setFilenamesOut()
{
    _FilenamesOut.clear();

    for (int aK=0;aK < _FilenamesIn.size();++aK)
    {
        QFileInfo fi(_FilenamesIn[aK]);

        _FilenamesOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + _postFix + ".tif");
    }
}

void cLoader::setFilenameOut(QString str)
{
    _FilenamesOut.clear();

    _FilenamesOut.push_back(str);
}

void cLoader::setPostFix(QString str)
{
    _postFix = str;
}

void cLoader::setSelectionFilename()
{
    _SelectionOut = _Dir.absolutePath() + QDir::separator() + "SelectionInfos.xml";
}

Cloud* cLoader::loadCloud( string i_ply_file, int* incre )
{
    return Cloud::loadPly( i_ply_file, incre );
}

int	ByP=-1;
std::string MkFT;

void DoMkT()
{
    if (ByP)
    {
        std::string aSMkSr = g_externalToolHandler.get( "make" ).callName()+" all -f " + MkFT + string(" -j")+ToString(ByP)/*+" -s"*/;
        System(aSMkSr,true);
    }
}

void cLoader::loadImage(QString aNameFile , QImage* &aImg, QImage* &aImgMask)
{
    aImg = new QImage( aNameFile );

    QFileInfo fi(aNameFile);

    QString mask_filename = fi.path() + QDir::separator() + fi.completeBaseName() + "_Masq.tif";

    setFilenameOut(mask_filename);

    if (aImg->isNull())
    {
        Tiff_Im aTF= Tiff_Im::StdConvGen(aNameFile.toStdString(),3,false);

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

        aImg = new QImage(aSz.x, aSz.y, QImage::Format_RGB32);

        for (int y=0; y<aSz.y; y++)
        {
            for (int x=0; x<aSz.x; x++)
            {
                QColor col(aDataR[y][x],aDataG[y][x],aDataB[y][x]);

                aImg->setPixel(x,y,col.rgb());
            }
        }
    }

    *aImg = QGLWidget::convertToGLFormat( *aImg );

    if (QFile::exists(mask_filename))
    {

        aImgMask = new QImage( mask_filename );

        if (aImgMask->isNull())
        {
            Tiff_Im imgMask( mask_filename.toStdString().c_str() );

            if( imgMask.can_elise_use() )
            {
                int w = imgMask.sz().x;
                int h = imgMask.sz().y;

                delete aImgMask;
                aImgMask = new QImage( w, h, QImage::Format_Mono);
                aImgMask->fill(0);

                Im2D_Bits<1> aOut(w,h,1);
                ELISE_COPY(imgMask.all_pts(),imgMask.in(),aOut.out());

                for (int x=0;x< w;++x)
                    for (int y=0; y<h;++y)
                        if (aOut.get(x,y) == 1 )
                            aImgMask->setPixel(x,y,1);

                *aImgMask = QGLWidget::convertToGLFormat( *aImgMask );                                
            }
            else
            {
                QMessageBox::critical(NULL, "cLoader::loadMask","Cannot load mask image");
            }
        }
        else
            *aImgMask = QGLWidget::convertToGLFormat( *aImgMask );

    }
}

// File structure is assumed to be a typical Micmac workspace structure:
// .ply files are in /MEC folder and orientations files in /Ori- folder
// /MEC and /Ori- are in the main working directory (m_Dir)

CamStenope* cLoader::loadCamera(QString aNameFile)
{
    string DirChantier = (_Dir.absolutePath()+ QDir::separator()).toStdString();
    string filename    = aNameFile.toStdString();

    #ifdef _DEBUG
        cout << "DirChantier : " << DirChantier << endl;
        cout << "filename : "    << filename << endl;
    #endif

    QFileInfo fi(aNameFile);

    _FilenamesOut.push_back(fi.path() + QDir::separator() + fi.completeBaseName() + "_Masq.tif");

    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(DirChantier);

    return CamOrientGenFromFile(filename.substr(DirChantier.size(),filename.size()),anICNM);
}

//****************************************
//   cEngine

cEngine::cEngine():    
    _Loader(new cLoader),
    _Data(new cData),
    _Gamma(1.f)
{}

cEngine::~cEngine()
{
    delete _Data;
    delete _Loader;

    for (int aK=0; aK<_vGLData.size();++aK)
        delete _vGLData[aK];
    _vGLData.clear();
}

void cEngine::loadClouds(QStringList filenames, int* incre)
{
    for (int i=0;i<filenames.size();++i)
    {
        _Data->addCloud(_Loader->loadCloud(filenames[i].toStdString(), incre));
    }

    _Data->getBB();
}

void cEngine::loadCameras(QStringList filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        _Data->addCamera(_Loader->loadCamera(filenames[i]));
    }

    _Data->getBB();
}

void cEngine::loadImages(QStringList filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        loadImage(filenames[i]);
    }

    _Loader->setFilenamesOut();
}

void  cEngine::loadImage(QString imgName)
{
    QImage* img, *mask;
    img = mask = NULL;

    _Loader->loadImage(imgName, img, mask);

    if (img !=NULL) _Data->PushBackImage(img);
    if (mask!=NULL) _Data->PushBackMask(mask);
#ifdef _DEBUG
    else cout << "mask null" << endl;
#endif
}

void cEngine::do3DMasks()
{
    CamStenope* pCam;
    Cloud *pCloud;
    Vertex vert;
    Pt2dr ptIm;

    for (int cK=0;cK < _Data->getNbCameras();++cK)
    {
        pCam = _Data->getCamera(cK);

        Im2D_BIN mask = Im2D_BIN(pCam->Sz(), 0);

        for (int aK=0; aK < _Data->getNbClouds();++aK)
        {
            pCloud  = _Data->getCloud(aK);

            for (int bK=0; bK < pCloud->size();++bK)
            {
                vert = pCloud->getVertex(bK);

                if (vert.isVisible())  //visible = selected in GLWidget
                {
                    Pt3dr pt(vert.getPosition());

                    if (pCam->PIsVisibleInImage(pt)) //visible = projected inside image
                    {
                        ptIm = pCam->Ter2Capteur(pt);
                        mask.set(floor(ptIm.x), floor(ptIm.y), 1);
                    }
                }
            }
        }

        string aOut = _Loader->getFilenamesOut()[cK].toStdString();
#ifdef _DEBUG
        printf ("Saving %s\n", aOut);
#endif

        Tiff_Im::CreateFromIm(mask, aOut);

#ifdef _DEBUG
        printf ("Done\n");
#endif
    }
}

void cEngine::doMaskImage(ushort idCur)
{
    QImage pMask = _vGLData[idCur]->getMask()->mirrored().convertToFormat(QImage::Format_Mono);

    if (!pMask.isNull())
    {
        QString aOut = _Loader->getFilenamesOut()[idCur];

        pMask.save(aOut);

		cFileOriMnt anOri;

        anOri.NameFileMnt()		= aOut.toStdString();
        anOri.NombrePixels()	= Pt2di(pMask.width(),pMask.height());
		anOri.OriginePlani()	= Pt2dr(0,0);
		anOri.ResolutionPlani() = Pt2dr(1.0,1.0);
		anOri.OrigineAlti()		= 0.0;
		anOri.ResolutionAlti()	= 1.0;
		anOri.Geometrie()		= eGeomMNTFaisceauIm1PrCh_Px1D;

        MakeFileXML(anOri, StdPrefix(aOut.toStdString()) + ".xml");
	}
	else
    {
        QMessageBox::critical(NULL, "cEngine::doMaskImage","No alpha channel!!!");
    }
}

void cEngine::saveMask(ushort idCur)
{
    if (getData()->getNbImages())
        doMaskImage(idCur);
    else
        do3DMasks();
}

void cEngine::saveSelectInfos(const QVector<selectInfos> &Infos)
{
    QDomDocument doc;

    QFile outFile(_Loader->getSelectionFilename());
    if (!outFile.open(QIODevice::WriteOnly)) return;

    QDomElement SI = doc.createElement("SelectionInfos");

    QDomText t;
    for (int i = 0; i < Infos.size(); ++i)
    {
        QDomElement SII            = doc.createElement("Item");
        QDomElement mvMatrixElem   = doc.createElement("ModelViewMatrix");
        QDomElement ProjMatrixElem = doc.createElement("ProjMatrix");
        QDomElement glViewportElem = doc.createElement("glViewport");
        QDomElement Mode           = doc.createElement("Mode");

        const selectInfos &SInfo = Infos[i];

        if ((SInfo.mvmatrix != NULL) && (SInfo.projmatrix != NULL) && (SInfo.glViewport != NULL))
        {
            QString text1, text2;

            text1 = QString::number(SInfo.mvmatrix[0], 'f');
            text2 = QString::number(SInfo.projmatrix[0], 'f');

            for (int aK=0; aK < 16;++aK)
            {
                text1 += " " + QString::number(SInfo.mvmatrix[aK], 'f');
                text2 += " " + QString::number(SInfo.projmatrix[aK], 'f');
            }

            t = doc.createTextNode(text1);
            mvMatrixElem.appendChild(t);

            t = doc.createTextNode(text2);
            ProjMatrixElem.appendChild(t);

            text1 = QString::number(SInfo.glViewport[0]) ;
            for (int aK=1; aK < 4;++aK)
                text1 += " " + QString::number(SInfo.glViewport[aK]);

            t = doc.createTextNode(text1);
            glViewportElem.appendChild(t);

            SII.appendChild(mvMatrixElem);
            SII.appendChild(ProjMatrixElem);
            SII.appendChild(glViewportElem);

            QVector <QPointF> pts = SInfo.poly;

            for (int aK=0; aK < pts.size(); ++aK)
            {
                QDomElement Point    = doc.createElement("Pt");
                QString str = QString::number(pts[aK].x(), 'f',1) + " "  + QString::number(pts[aK].y(), 'f',1);

                t = doc.createTextNode( str );
                Point.appendChild(t);
                SII.appendChild(Point);
            }

            t = doc.createTextNode(QString::number(SInfo.selection_mode));
            Mode.appendChild(t);

            SII.appendChild(Mode);

            SI.appendChild(SII);
        }
        else
            cerr << "saveSelectInfos: null matrix";

    }

    doc.appendChild(SI);

    QTextStream content(&outFile);
    content << doc.toString();
    outFile.close();

#ifdef _DEBUG
        printf ( "File saved in: %s\n", _Loader->getSelectionFilename().toStdString().c_str());
#endif
}

void cEngine::applyGammaToImage(int aK)
{
    _Data->applyGammaToImage(aK, _Gamma);
}

void cEngine::unloadAll()
{
    _Data->clearClouds();
    _Data->clearCameras();
    _Data->clearImages();
    _Data->clearMasks();
    _Data->reset();

    for (int aK=0; aK<_vGLData.size();++aK)
        delete _vGLData[aK];
    _vGLData.clear();
}

void cEngine::setGLData()
{
    _vGLData.clear();

    for (int aK = 0; aK < _Data->getNbImages();++aK)
    {

        cGLData *glData = new cGLData(_Data->getImage(aK),_Data->getMask(aK));

        // TODO _Data->addMask(theData->pQMask) ne prend pas en compte l'ordre des images
        if(_Data->getMask(aK) == NULL) _Data->PushBackMask(glData->pQMask);

        _vGLData.push_back(glData);
    }

    if (_Data->is3D())
        _vGLData.push_back(new cGLData(_Data));

}

cGLData* cEngine::getGLData(int WidgetIndex)
{
    if ((_vGLData.size() > 0) && (WidgetIndex < _vGLData.size()))
        return _vGLData[WidgetIndex];
    else
        return NULL;
}

//********************************************************************************

cGLData::cGLData():
    _diam(1.f){}

cGLData::cGLData(QImage *image, QImage *mask):
    pBall(NULL),
    pAxis(NULL),
    pBbox(NULL)
{
    // TODO a factoriser dans maskedimage!!
    //

    if(mask == NULL)
    {
        pQMask = new QImage(image->size(),QImage::Format_Mono);
        *pQMask = QGLWidget::convertToGLFormat( *pQMask );
        pQMask->fill(Qt::white);
        maskedImage._m_newMask = true;
    }
    else
       pQMask = mask;

    maskedImage._m_mask = new cImageGL();
    maskedImage._m_image = new cImageGL();

    maskedImage._m_mask->PrepareTexture(pQMask);
    maskedImage._m_image->PrepareTexture(image);

}

cGLData::cGLData(cData *data):
    _diam(1.f)
{
    for (int aK = 0; aK < data->getNbClouds();++aK)
    {
        Cloud *pCloud = data->getCloud(aK);
        Clouds.push_back(pCloud);
        pCloud->setBufferGl();
    }

    Pt3dr center = data->getCenter();
    float scale = data->m_diam / 1.5f;

    pBall = new cBall(center, scale);
    pAxis = new cAxis(center, scale);
    pBbox = new cBBox(center, scale, data->m_min, data->m_max);

    for (int i=0; i< data->getNbCameras(); i++)
    {
        cCam *pCam = new cCam(data->getCamera(i), scale);

        Cams.push_back(pCam);
    }

    setScale(data->getScale());
    setCenter(data->getCenter());
}

cGLData::~cGLData()
{

    if(maskedImage._m_image != NULL) delete maskedImage._m_image;
    if(maskedImage._m_mask != NULL) delete maskedImage._m_mask;

    for (int aK = 0; aK< Cams.size(); ++aK) delete Cams[aK];
    //qDeleteAll(Cams);
    Cams.clear();

    if(pBall != NULL) delete pBall;
    if(pAxis != NULL) delete pAxis;
    if(pBbox != NULL) delete pBbox;

   //pas de delete des pointeurs dans Clouds c'est Data qui s'en charge
    Clouds.clear();
}

void cGLData::InsertPointPolygon()
{
    if ((m_polygon.size() >=2) && m_dihedron.size() && m_polygon.isClosed())
    {
        int idx = -1;

        for (int i=0;i<m_polygon.size();++i)
        {
            if (m_polygon[i] == m_dihedron[0]) idx = i;
        }

        if (idx >=0) m_polygon.insert(idx+1, m_dihedron[1]);
    }

    m_dihedron.clear();
}

void cGLData::RemoveClosestPoint(QPointF pos, bool &lastAction)
{
    int idx = m_polygon.idx();
    if ((idx >=0)&&(idx<m_polygon.size())&&m_polygon.isClosed())
    {
        m_polygon.remove(idx);   // remove closest point

        m_polygon.findClosestPoint(pos);

        if (m_polygon.size() < 3)
            m_polygon.setClosed(false);

        lastAction = true;
    }
    else if (m_polygon.size() == 2)
    {
        m_polygon.remove(1);
        m_polygon.setClosed(false);
    }
    else // close polygon
        m_polygon.close();
}

void cGLData::AddPoint(QPointF pos)
{
    if (m_polygon.size() >= 1)
        m_polygon[m_polygon.size()-1] = pos;

    m_polygon.add(pos);
}

void cGLData::FinalMovePoint(QPointF pos)
{
    int idx = m_polygon.idx(); // index du point selectionné
    if ((m_polygon.click() >=1) && (idx>=0) && m_dihedron.size()) //  fin de deplacement point
    {
        m_polygon[idx] = m_dihedron[1];

        m_dihedron.clear();
        m_polygon.resetClick();
    }

    // TODO refactoriser
    if ((m_polygon.click() >=1) && m_polygon.isClosed()) // recherche de points le plus proche
    {
        m_polygon.findClosestPoint(pos);
    }

}

void cGLData::RefreshHelperPolygon(QPointF pos, bool insertMode, bool &lastAction)
{
    int nbVertex =m_polygon.size();

    if(m_polygon.isOpened())
    {
        if (nbVertex == 1)     // add current mouse position to polygon (dynamic display)
           m_polygon.add(pos);
        else if ((nbVertex == 2) && lastAction)
           m_polygon.add(pos);
        else if (nbVertex > 1) // replace last point by the current one
           m_polygon[nbVertex-1] = pos;

        lastAction = false;
    }
    else if(nbVertex)                       // move vertex or insert vertex (dynamic display) en court d'opération
    {
        if (insertMode )                    // INSERT POINT POLYGON

           m_polygon.fillDihedron(pos,m_dihedron);

        else if (m_polygon.click() == 1)    // MOVE POINT POLYGON

           m_polygon.fillDihedron2(pos,m_dihedron);
        else                                // SELECT CLOSEST POINT POLYGON

           m_polygon.findClosestPoint(pos);
    }
}

void cGLData::draw()
{
    enableOptionLine();

    for (int i=0; i<Clouds.size();i++)
        Clouds[i]->draw();

    if (pBall->isVisible())
        pBall->draw();
    else if (pAxis->isVisible())
        pAxis->draw();

    pBbox->draw();

    //cameras
    for (int i=0; i< Cams.size();i++) Cams[i]->draw();

    disableOptionLine();
}

//********************************************************************************

ViewportParameters::ViewportParameters()
    : m_zoom(1.f)
    , m_PointSize(1)
    , m_LineWidth(1.f)
    , m_speed(2.f)
{
    m_translationMatrix[0] = m_translationMatrix[1] = m_translationMatrix[2] = 0.f;
}

ViewportParameters::ViewportParameters(const ViewportParameters& params)
    : m_zoom(params.m_zoom)
    , m_PointSize(params.m_PointSize)
    , m_LineWidth(params.m_LineWidth)
{
    m_translationMatrix[0] = params.m_translationMatrix[0];
    m_translationMatrix[1] = params.m_translationMatrix[1];
    m_translationMatrix[2] = params.m_translationMatrix[2];
}

ViewportParameters::~ViewportParameters(){}

ViewportParameters& ViewportParameters::operator =(const ViewportParameters& par)
{
    if (this != &par)
    {
        m_zoom = par.m_zoom;
        m_PointSize = par.m_PointSize;

        m_translationMatrix[0] = par.m_translationMatrix[0];
        m_translationMatrix[1] = par.m_translationMatrix[1];
        m_translationMatrix[2] = par.m_translationMatrix[2];
        m_LineWidth = par.m_LineWidth;
    }

    return *this;
}

void ViewportParameters::reset()
{
    m_zoom = m_LineWidth = 1.f;
    m_PointSize = 1;

    m_translationMatrix[0] = m_translationMatrix[1] = m_translationMatrix[2] = 0.f;
}

void ViewportParameters::ptSizeUp(bool up)
{
    if (up)
        m_PointSize++;
    else
        m_PointSize--;

    if (m_PointSize == 0)
        m_PointSize = 1;

    glPointSize((GLfloat) m_PointSize);
}


