#include "GLWidget.h"

GLWidget::GLWidget(int idx,  const QGLWidget *shared) : QGLWidget(QGLFormat(QGL::SampleBuffers),NULL,shared)
  , m_interactionMode(TRANSFORM_CAMERA)
  , m_bFirstAction(true)
  , m_GLData(NULL)
  , m_bDisplayMode2D(false)
  , _vp_Params(ViewportParameters())
  , _frameCount(0)
  , _previousTime(0)
  , _currentTime(0)
  , _messageManager(this)
  , _widgetId(idx)
{
    _matrixManager.resetAllMatrix();

    _time.start();

    setAcceptDrops(true);           //drag & drop handling

    setMouseTracking(true);

    setOption(cGLData::OpShow_Mess);

    _contextMenu.createContextMenuActions();
}

void GLWidget::resizeGL(int width, int height)
{

    if (width==0 || height==0) return;

    _matrixManager.setGLViewport(0,0,width, height);
    _messageManager.wh(width, height);

    zoomFit();
}

//-------------------------------------------------------------------------
// Computes the frames rate
//-------------------------------------------------------------------------

void GLWidget::computeFPS(MessageToDisplay &dynMess)
{
    float       fps;

    //  Increase frame count
    _frameCount++;

    _currentTime = _time.elapsed();

    //  Compute elapsed time
    int deltaTime = _currentTime - _previousTime;

    if(deltaTime > 1000)
    {
        //  compute the number of frames per second
        fps = _frameCount * 1000.f / deltaTime;

        //  Set time
        _previousTime = _currentTime;

        //  Reset frame count
        _frameCount = 0;


        if (fps > 1e-3)
            dynMess.message = "fps: " + QString::number(fps,'f',1);
    }
}

ContextMenu* GLWidget::contextMenu()
{
    return &_contextMenu;
}

void GLWidget::setGLData(cGLData * aData, bool showMessage, bool showCams, bool doZoom, bool resetPoly)
{
    if (aData != NULL)
    {
        if(_widgetId != -1 && m_GLData && !m_GLData->isImgEmpty())
            m_GLData->glImage().deleteTextures();

        m_GLData = aData;

        if(_widgetId != -1 && m_GLData && !m_GLData->isImgEmpty())
            m_GLData->glImage().prepareTextures();


//        if(!m_GLData->isImgEmpty())
//        {
//            int rglError = (int)m_GLData->glImage().cMaskedImageGL::_m_image->glError();
//            if(rglError)
//            {
//                QMessageBox msgBox;
//                QString nameImageT = m_GLData->glImage().cObjectGL::name();
//                QString messageError = nameImageT + QString(" Error ") + QString::number(rglError);
//                msgBox.setText(messageError);
//                msgBox.exec();

//                qDebug() << m_GLData->glImage().cMaskedImageGL::_m_mask->glError();
//            }
//        }

        m_bDisplayMode2D = !m_GLData->isImgEmpty();
        m_bFirstAction   =  m_GLData->isNewMask();

        _contextMenu.setPolygon( m_GLData->currentPolygon() );

        _matrixManager.setSceneTopo(getGLData()->getBBoxCenter(),getGLData()->getBBoxMaxSize());

        resetView(doZoom, showMessage, showCams, true, resetPoly);
    }
}

cPolygon *GLWidget::polygon(int id){ return m_GLData->polygon(id); }

cPolygon *GLWidget::polygon(){

    return m_GLData ? m_GLData->currentPolygon() : NULL;
}

void GLWidget::addGlPoint(QPointF pt, cOneSaisie* aSom, QPointF pt1, QPointF pt2, bool highlight)
{
    QString name(aSom->NamePt().c_str());
    cPoint point(pt,name,true,aSom->Etat());
    point.setDiameter(_params->getPointDiameter() * 0.01);

    point.setHighlight(highlight);

    if (pt1 != QPointF(0.f,0.f)) point.setEpipolar(pt1, pt2);

    getGLData()->currentPolygon()->add(point);
}

void GLWidget::setTranslation(Pt3d<double> trans)
{
    _matrixManager.resetTranslationMatrix(trans);
}

bool GLWidget::imageLoaded()
{
    return hasDataLoaded() &&  m_bDisplayMode2D;
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //gradient color background
    cImageGL::drawGradientBackground(vpWidth(), vpHeight(), _BGColor0, !hasDataLoaded() || imageLoaded() ? _BGColor0 : _BGColor1);

    glClear(GL_DEPTH_BUFFER_BIT);

    if (hasDataLoaded())
    {
        _matrixManager.applyAllTransformation(m_bDisplayMode2D,m_lastClickZoom,_vp_Params.m_zoom);

        m_GLData->draw();

        overlay();

        if (_messageManager.drawMessages() && !m_bDisplayMode2D)
            computeFPS(_messageManager.LastMessage());
    }

    _messageManager.draw();
}

int GLWidget::getWindowMeanValue(QPoint pos, int r)
{
    QImage gIma = grabFrameBuffer();

    int kP  = 0;
    int cP  = 0;

    for (int x = max(0,pos.x()-r); x < min(gIma.width()-1,pos.x()+r); ++x)
        for (int y = max(0,pos.y()-r); y < min(gIma.height()-1,pos.y()+r); ++y)
    {
        cP  += qGray(gIma.pixel(x, y));
         ++kP;
    }

    cP /= kP;

    return cP;
}

void GLWidget::overlay()
{
    if (hasDataLoaded() && (m_bDisplayMode2D || (m_interactionMode == SELECTION)) )
    {
        if (_widgetId < 0)
        {
            GLint       glViewport[4];
            glGetIntegerv(GL_VIEWPORT, glViewport);

            //todo remplacer par la croix
            int cP = getWindowMeanValue(QPoint( (int)glViewport[2]/2, (int)glViewport[3]/2 ) );

            m_GLData->drawCenter(cP<128);
        }

        if(m_interactionMode == SELECTION)
            _matrixManager.setMatrixDrawViewPort();

        for (int i = 0; i < m_GLData->polygonCount(); ++i)
        {
            cPolygon* polyg = polygon(i);

            if (polyg)
            {
                polyg->draw();

                if (polyg->bShowNames())
                {
                    for (int aK=0; aK < polyg->size();++aK)
                    {
                        cPoint pt = polyg->operator [](aK);

                        if (pt.showName() && (pt.name() != ""))
                        {
                            QPointF wPt = _matrixManager.ImageToWindow( pt,_vp_Params.m_zoom) + QPointF(10.f,-5.f);

                            _messageManager.glRenderText(pt.name(),wPt, pt.isSelected() ? Qt::blue : Qt::white);
                        }
                    }
                }
            }
        }

        if(m_interactionMode == SELECTION)
            glPopMatrix();
    }
}

void GLWidget::setInteractionMode(int mode, bool showmessage, bool showcams)
{
    m_interactionMode = mode;

    resetView(false, showmessage, showcams, false);
}

void GLWidget::setView(VIEW_ORIENTATION orientation)
{
    if (hasDataLoaded())
       _matrixManager.setView(orientation,m_GLData->getBBoxCenter());
}

void GLWidget::centerViewportOnImagePosition(QPointF pt, float zoom)
{

    m_lastClickZoom = QPoint((int) (vpWidth()*.5f), (int) (vpHeight()*.5f));

    _matrixManager.resetMatrixProjection(-pt.x(), -pt.y());

    if(zoom > 0.f)
        setZoom(zoom);

    update();
}

void GLWidget::lineThicknessChanged(float val)
{
    if (hasDataLoaded())
    {
        polygon()->setLineWidth(val);

        update();
    }
}

void GLWidget::gammaChanged(float val)
{
    if (hasDataLoaded())
    {
        m_GLData->glImage()._m_image->setGamma(val);
        update();
    }
}

void GLWidget::pointDiameterChanged(float val)
{
    if (hasDataLoaded())
    {
        polygon()->setPointSize(val);

        update();
    }
}

void GLWidget::selectionRadiusChanged(int val)
{
    if (hasDataLoaded())
    {
        polygon()->setRadius(val);
    }
}

void GLWidget::shiftStepChanged(float val)
{
    if (hasDataLoaded())
    {
        polygon()->setShiftStep(val);
    }
}

void GLWidget::forceGray(bool val)
{
    if (hasDataLoaded())
    {
        //TODO
        //reload in gray
        update();
    }
}

void GLWidget::showMasks(bool val)
{
    if (hasDataLoaded())
    {
        m_GLData->glImage().showMask(val);
        update();
    }
}

void GLWidget::setParams(cParameters* aParams)
{
    _params = aParams;

    if(polygon())
        polygon()->setParams(aParams);
}

void GLWidget::setZoom(float val)
{
    if (imageLoaded())  zoomClip( val );

    _vp_Params.m_zoom = val;

    if(imageLoaded() && _messageManager.drawMessages())
        _messageManager.GetLastMessage()->message = QString::number(_vp_Params.m_zoom*100,'f',1) + "%";

    update();
}

void GLWidget::zoomFit()
{
    if (hasDataLoaded())
    {
        if(m_bDisplayMode2D)
        {
            QPointF imCenter(imWidth()*.5f, imHeight()*.5f);
            centerViewportOnImagePosition(imCenter);

            float rw = (float) (1.05f*imWidth())  / vpWidth();
            float rh = (float) (1.05f*imHeight()) / vpHeight();

            if(rw>rh)
                setZoom(1.f/rw); //orientation landscape
            else
                setZoom(1.f/rh); //orientation portrait
        }

        else
            setZoom(m_GLData->getBBoxMaxSize());
    }
}

void GLWidget::selectPoint(QString namePt)
{
    if(polygon())
        polygon()->selectPoint(namePt);
    update();
}

void GLWidget::zoomFactor(int percent)
{
    if (m_bDisplayMode2D)
    {
        m_lastClickZoom = m_lastPosWindow;
        setZoom(0.01f * percent);
    }
    else if (hasDataLoaded())
        setZoom(m_GLData->getBBoxMaxSize() / (float) percent * 100.f);
}

void GLWidget::setCursorShape(QPointF pos, QPointF mPos)
{
    QCursor c = cursor();

    if ((mPos.x() > 0) && (mPos.y() > 0) && (mPos.x() < vpWidth()) && (mPos.y() < vpHeight())
    &&  imageLoaded() && !polygon()->isLinear() && isPtInsideIm(pos) && (_widgetId >=0) )
    {
        QPixmap cuCross(":/MM/Icons/images/cross_cursor.png");

        if( getWindowMeanValue(QPoint((int) mPos.x(), (int)mPos.y())) < 128)
        {
            QImage image = cuCross.toImage();
            image.invertPixels();

            cuCross = cuCross.fromImage(image);
        }

        c = QCursor(cuCross, cuCross.width()/2, cuCross.height()/2);
    }
    else
        c.setShape(Qt::ArrowCursor);


    setCursor(c);
}

void GLWidget::Select(int mode, bool saveInfos)
{
    if (hasDataLoaded())
    {
        cPolygon polyg = *polygon();

        if(mode <= ADD_OUTSIDE)
        {
           if (polyg.size() == 0)
           {
               QMessageBox::warning(this,tr("Warning"), tr("Draw a polygon first\n\nLeft clic:\tadd vertex\nRight clic:\tclose polygon"));
               return;
           }
           else  if ((polyg.size() < 3) || (!polyg.isClosed()))
               return;
        }

        if (m_bDisplayMode2D)
            m_GLData->editImageMask(mode,polyg,m_bFirstAction);
        else
            m_GLData->editCloudMask(mode,polyg,m_bFirstAction,_matrixManager);

        if (mode <= ADD_OUTSIDE) m_bFirstAction = false;

        if (saveInfos) //  TODO ne marche pas avec le switch y/z
        {
            selectInfos info(polygon()->getVector(),mode);

            _matrixManager.exportMatrices(info);

            _historyManager.push_back(info);
        }

        emit maskEdited();

        m_GLData->clearPolygon();

        update();
    }
}

void GLWidget::applyInfos()
{
    if (hasDataLoaded())
    {
        int actionIdx = _historyManager.getActionIdx();

        QVector <selectInfos> vInfos = _historyManager.getSelectInfos();

        if (actionIdx < 0 || actionIdx > vInfos.size()) return;

        for (int aK = 0; aK < actionIdx ; aK++)
        {
            selectInfos &infos = vInfos[aK];

            cPolygon * poly = new cPolygon();
            poly->setVector(infos.poly);
            poly->setClosed(true);
            poly->setPointSize(_params->getPointDiameter());

            m_GLData->setPolygon(0, poly);

            if (!m_bDisplayMode2D)

                _matrixManager.importMatrices(infos);

            Select(infos.selection_mode, false);
        }
    }
}

void GLWidget::setOption(QFlags<cGLData::Option> option, bool show)
{
    if (hasDataLoaded()) m_GLData->setOption(option,show);

    if( option & cGLData::OpShow_Mess)
        _messageManager.constructMessagesList(show, m_interactionMode, m_bDisplayMode2D, hasDataLoaded(), getZoom());

    update();
}

void GLWidget::reset()
{
    _vp_Params.reset();
    _historyManager.reset();

    m_bFirstAction = true;

    m_GLData = NULL;

    resetView();
}

void GLWidget::resetView(bool zoomfit, bool showMessage, bool showCams, bool resetMatrix, bool resetPoly)
{

    if (resetMatrix)
        _matrixManager.resetAllMatrix( hasDataLoaded() ? m_GLData->getBBoxCenter() : Pt3dr(0.f,0.f,0.f) );

    if (hasDataLoaded() && resetPoly) m_GLData->clearPolygon();

    setOption(cGLData::OpShow_Mess, showMessage);
    setOption(cGLData::OpShow_Cams, showCams);

    if (!m_bDisplayMode2D)
    {
        setOption(cGLData::OpShow_Ball, m_interactionMode == TRANSFORM_CAMERA);

        setOption(cGLData::OpShow_Axis, false);

        setOption(cGLData::OpShow_Grid, m_interactionMode == TRANSFORM_CAMERA);

        if (m_interactionMode == SELECTION)

            setOption(cGLData::OpShow_BBox | cGLData::OpShow_Cams, false);
    }

    if (zoomfit)
        zoomFit(); //update already done in zoomFit
    else
        update();
}


void GLWidget::wheelEvent(QWheelEvent* event)
{
    if ((m_interactionMode == SELECTION)&&(!m_bDisplayMode2D))
    {
        event->ignore();
        return;
    }

    m_lastClickZoom = event->pos();

#if ELISE_QT_VERSION == 5
    setZoom(_vp_Params.m_zoom*pow(1.1f,event->angleDelta().y() / 70.0f ));
#else
    setZoom(_vp_Params.m_zoom*pow(1.1f,event->delta() / 70.0f ));
#endif
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    if(hasDataLoaded())
    {
        m_lastPosWindow = event->pos();

        m_lastPosImage =  m_bDisplayMode2D ? _matrixManager.WindowToImage(m_lastPosWindow, _vp_Params.m_zoom) : QPointF(m_lastPosWindow.x(),_matrixManager.vpHeight() - m_lastPosWindow.y());

        if (event->button() == Qt::LeftButton)
        {
            if (m_bDisplayMode2D || (m_interactionMode == SELECTION))
            {

                if(!polygon()->isClosed())             // ADD POINT

                    polygon()->addPoint(m_lastPosImage);

                else if (polygon()->isLinear() && (event->modifiers() & Qt::ShiftModifier)) // INSERT POINT

                    polygon()->insertPoint();

                else if (polygon()->getSelectedPointIndex() != -1) // SELECT POINT

                    polygon()->setPointSelected();

                else if (!polygon()->isLinear() && isPtInsideIm(m_lastPosImage))

                    emit addPoint(m_lastPosImage);

            }
            else
                _matrixManager.handleRotation(event->pos());
        }
        else if (event->button() == Qt::RightButton && polygon())
        {
            if (polygon()->isLinear())
            {
                if (event->modifiers() & Qt::ControlModifier)

                    polygon()->removeLastPoint();

                else

                    polygon()->removeNearestOrClose(m_lastPosImage);
            }
        }
        else if (event->button() == Qt::MiddleButton)

            m_lastClickZoom = m_lastPosWindow;
    }
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if ( event->button() == Qt::LeftButton && hasDataLoaded())
    {
        m_lastPosWindow = event->pos();

        m_lastPosImage =  m_bDisplayMode2D ? _matrixManager.WindowToImage(m_lastPosWindow, _vp_Params.m_zoom) : m_lastPosWindow;

        int idMovePoint = polygon() ? polygon()->finalMovePoint() : -1; //ne pas factoriser

        if(polygon())
            polygon()->findNearestPoint(m_lastPosImage);

        update();

        emit movePoint(idMovePoint);
    }
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (hasDataLoaded())
    {

#if ELISE_QT_VERSION == 5
        QPointF mPos = event->localPos();
#else
        QPointF mPos = event->posF();
#endif

        QPointF pos = m_bDisplayMode2D ?  _matrixManager.WindowToImage(mPos, _vp_Params.m_zoom) : mPos;

        if ( event->buttons() != Qt::MiddleButton )
            setCursorShape(pos,mPos);

        if (m_bDisplayMode2D)

            m_lastMoveImage = pos;


        if (m_bDisplayMode2D || (m_interactionMode == SELECTION))
        {
            if (polygon()->isSelected())                    // MOVE POLYGON
            {
                //TODO: a verifier => y inversé en 3D - OK en 2D
                QPointF translation = m_bDisplayMode2D ? _matrixManager.WindowToImage(m_lastPosWindow, _vp_Params.m_zoom) : m_lastPosWindow;
                polygon()->translate(pos - translation);
            }
            else if (!(event->buttons() == Qt::MiddleButton))// REFRESH HELPER POLYGON
            {
                int id = polygon()->getSelectedPointIndex();

                bool insertMode = polygon()->isLinear() ? (event->modifiers() & Qt::ShiftModifier) : event->type() == QMouseEvent::MouseButtonPress;

                if(m_interactionMode == SELECTION)
                    polygon()->refreshHelper( QPointF(pos.x(),_matrixManager.vpHeight() - pos.y()), insertMode, 1.f);
                else
                    polygon()->refreshHelper(pos, insertMode, _vp_Params.m_zoom);

                if (polygon()->size() && m_bDisplayMode2D)

                    emit newRectanglePosition(polygon()->transfoTerrain(m_GLData->glImage()));

                if((id != polygon()->getSelectedPointIndex()))

                    emit selectPoint(polygon()->getSelectedPointIndex());

            }
        }

        if (m_interactionMode == TRANSFORM_CAMERA)
        {
            QPointF dPWin = QPointF(event->pos() - m_lastPosWindow);

            if (event->buttons())
            {
                Pt3dr r(0,0,0);

                if (event->buttons() == Qt::LeftButton)               // ROTATION X et Y
                {
                    r.x = dPWin.y() / vpWidth();
                    r.y = dPWin.x() / vpHeight();
                }
                else if (event->buttons() == Qt::MiddleButton)
                {
                    if (event->modifiers() & Qt::ShiftModifier)         // ZOOM VIEW

                        setZoom(_vp_Params.changeZoom(dPWin.y()));

                    else                                                // TRANSLATION VIEW
                    {
                        //QPointF dp = m_bDisplayMode2D ? pos - m_lastPosImage : QPointF(dPWin .x(),-dPWin .y())/vpWidth()*getGLData()->getBBoxMaxSize()*_vp_Params.m_zoom/2.f;

                        QPointF dp = m_bDisplayMode2D ? pos - m_lastPosImage : _matrixManager.screen2TransABall(dPWin);
                        _matrixManager.translate(dp.x(),dp.y(),0.0);
                    }
                }
                else if (event->buttons() == Qt::RightButton)           // ROTATION Z
                    r.z = (float)dPWin.x() / vpWidth();

                _matrixManager.rotateArcBall(r.y, r.x, r.z, _vp_Params.m_speed * 2.f);
            }
        }

        if (event->buttons() != Qt::MiddleButton)  //pour eviter le changement de label_ImagePosition_2 en mode translation
            emit newImagePosition( m_lastMoveImage );

        m_lastPosWindow = event->pos();

        update();
    }
}

void GLWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
    if (hasDataLoaded() && m_interactionMode == TRANSFORM_CAMERA && m_GLData->cloudCount() )
    {
#if ELISE_QT_VERSION == 5
        QPointF pos = event->localPos();
#else
        QPointF pos = event->posF();
#endif

        if (m_GLData->position2DClouds(_matrixManager,pos))
           update();
    }
}

void GLWidget::contextMenuEvent(QContextMenuEvent * event)
{
    QMenu menu(this);

    if (hasDataLoaded())
    {
        if (polygon()->findNearestPoint(m_lastPosImage, polygon()->getRadius()/getZoom()))
        {
            menu.addAction(_contextMenu._validate   );
            menu.addAction(_contextMenu._dubious    );
            menu.addAction(_contextMenu._refuted    );
            menu.addAction(_contextMenu._noSaisie   );
            menu.addAction(_contextMenu._highLight  );
            menu.addSeparator();
            menu.addAction(_contextMenu._rename     );
            menu.addSeparator();
            menu.addAction(_contextMenu._ThisP );
        }
        else
        {
            //menu.setWindowTitle(tr("Switch"));

            //menu.setWindowFlags(Qt::Tool | Qt::WindowTitleHint | Qt::WindowStaysOnTopHint);

            menu.addAction( _contextMenu._AllW  );
            menu.addAction( _contextMenu._RollW );
            menu.addAction( _contextMenu._ThisW );
            menu.addAction( _contextMenu._ThisP );
        }

        _contextMenu.setPos(m_lastPosImage);

        menu.exec(event->globalPos());

        polygon()->resetSelectedPoint();

        emit selectPoint(-1);
    }
}

void GLWidget::enterEvent(QEvent *event)
{
    // TODO : pourquoi
    // peut etre capter les mvts de souris en dehors de la fenetre
    setFocus(Qt::ActiveWindowFocusReason);
    setFocusPolicy(Qt::StrongFocus);

    emit overWidget(this);
}

void GLWidget::movePointWithArrows(QKeyEvent* event)
{
    QPointF tr(0.f, 0.f);
    float shift = polygon()->getShiftStep();

    if (event->modifiers() & Qt::AltModifier)
        shift *= 10.f;

    switch(event->key())
    {
    case Qt::Key_Up:
        tr.setY(shift);
        break;
    case Qt::Key_Down:
        tr.setY(-shift);
        break;
    case Qt::Key_Left:
        tr.setX(-shift);
        break;
    case Qt::Key_Right:
        tr.setX(shift);
        break;
    default:
        break;
    }

    cPoint pt = polygon()->translateSelectedPoint(tr);

    emit movePoint(polygon()->getSelectedPointIndex());

    //polygon()->helper()->clear();

    polygon()->findNearestPoint(pt, 400000.f);
}


void GLWidget::keyPressEvent(QKeyEvent* event)
{
    if(event->modifiers().testFlag(Qt::ControlModifier))
    {
        if(event->key() == Qt::Key_2)    zoomFactor(50);
        else if(event->key() == Qt::Key_4)    zoomFactor(25);
    }
    else
    {
        if (hasDataLoaded())
        {
            switch(event->key())
            {
            case Qt::Key_Delete:
                if (polygon())
                {
                    emit removePoint(eEPI_Disparu, m_GLData->currentPolygon()->getSelectedPointIndex());
                    polygon()->removeSelectedPoint();
                }
                break;
            case Qt::Key_Escape:
                if (polygon() && polygon()->isLinear()) m_GLData->clearPolygon();
                break;
            case Qt::Key_1:
                zoomFactor(100);
                break;
            case Qt::Key_2:
                zoomFactor(200);
                break;
            case Qt::Key_4:
                zoomFactor(400);
                break;
            case Qt::Key_9:
                zoomFit();
                break;
            case Qt::Key_G:
                if(m_bDisplayMode2D)
                {
                    m_GLData->glImage()._m_image->incGamma(0.2f);
                    emit gammaChangedSgnl(m_GLData->glImage()._m_image->getGamma());
                }
                break;
            case Qt::Key_H:
                if(m_bDisplayMode2D)
                {
                    m_GLData->glImage()._m_image->incGamma(-0.2f);
                    emit gammaChangedSgnl(m_GLData->glImage()._m_image->getGamma());
                }
                break;
            case Qt::Key_J:
                if(m_bDisplayMode2D)
                {
                    m_GLData->glImage()._m_image->setGamma(1.f);
                    emit gammaChangedSgnl(1.f);
                }
                break;
            case Qt::Key_Plus:
                if (m_bDisplayMode2D)
                {
                    m_lastClickZoom = m_lastPosWindow;
                    setZoom(_vp_Params.m_zoom*1.5f);
                }
                else
                    _vp_Params.ptSizeUp(true);
                break;
            case Qt::Key_Minus:
                if (m_bDisplayMode2D)
                {
                    m_lastClickZoom = m_lastPosWindow;
                    setZoom(_vp_Params.m_zoom/1.5f);
                }
                else
                    _vp_Params.ptSizeUp(false);
                break;
            case Qt::Key_W:
                setCursor(Qt::SizeAllCursor);
                if(polygon())
                {
                    polygon()->helper()->clear();
                    polygon()->setSelected(true);
                }
                break;
            case Qt::Key_Up:
            case Qt::Key_Down:
            case Qt::Key_Left:
            case Qt::Key_Right:
                movePointWithArrows(event);
                break;
            default:
                event->ignore();
                break;
            }
        }
    }

   update();
}

void GLWidget::keyReleaseEvent(QKeyEvent* event)
{
    if(hasDataLoaded() && (polygon()))
    {
        if (event->key() == Qt::Key_Shift )
        {
            polygon()->helper()->clear();
            polygon()->resetSelectedPoint();
        }

        polygon()->setSelected(false);

        update();
    }
    setCursor(Qt::ArrowCursor);
}

void GLWidget::dragEnterEvent(QDragEnterEvent *event)
{
    const QMimeData* mimeData = event->mimeData();

    if (mimeData->hasFormat("text/uri-list"))
        event->acceptProposedAction();
}

void GLWidget::dropEvent(QDropEvent *event)
{
    const QMimeData* mimeData = event->mimeData();

    if (mimeData->hasFormat("text/uri-list")) // TODO peut etre deplacer factoriser la gestion de drop fichier!!!
    {
        QByteArray data = mimeData->data("text/uri-list");
        QStringList fileNames = QUrl::fromPercentEncoding(data).split(QRegExp("\\n+"),QString::SkipEmptyParts);

        for (int i=0;i<fileNames.size();++i)
        {
            fileNames[i] = fileNames[i].trimmed();

#if defined(_WIN32) || defined(WIN32)
            fileNames[i].remove("file:///");
#else
            fileNames[i].remove("file://");
#endif
        }

        if (!fileNames.empty())
            emit filesDropped(fileNames, true);

        setFocus();

        event->acceptProposedAction();
    }

    event->ignore();
}

