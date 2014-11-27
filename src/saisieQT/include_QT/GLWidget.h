#ifndef _GLWIDGET_H
#define _GLWIDGET_H

#include "Engine.h"
#include "ContextMenu.h"

class GLWidget : public QGLWidget
{
    Q_OBJECT

public:

    //! Default constructor
    GLWidget(int idx, const QGLWidget *shared);

    //! Destructor
    ~GLWidget(){}

    //! States if data (cloud, camera or image) is loaded
    bool hasDataLoaded(){ return (m_GLData != NULL); }

    //! Sets camera to a predefined view (top, bottom, etc.)
    void setView(VIEW_ORIENTATION orientation);

    //! Get current zoom
    float getZoom();

    void zoomFit();

    void zoomFactor(int percent);

    //! Switch between move mode and selection mode (only in 3D)
    void setInteractionMode(int mode, bool showmessage, bool showcams);

    bool getInteractionMode(){return m_interactionMode;}

    void setOption(QFlags<cGLData::Option> option, bool show = true);

    //! Apply selection to data
    void Select(int mode, bool saveInfos = true);

    void applyInfos();

    //! Avoid all past actions
    void reset();

    //! Reset view
    void resetView(bool zoomfit = true, bool showMessage = true, bool showCams = true, bool resetMatrix = true, bool resetPoly = true);

    ViewportParameters* getParams()         { return &_vp_Params;      }
    HistoryManager*     getHistoryManager() { return &_historyManager; }
    cMessages2DGL*      getMessageManager() { return &_messageManager; }
    MatrixManager*      getMatrixManager()  { return &_matrixManager;  }

    void        setGLData(cGLData* aData, bool showMessage = true, bool showCams = true, bool doZoom = true, bool resetPoly = true);
    cGLData*    getGLData(){ return m_GLData; }

    void setBackgroundColors(QColor const &col0, QColor const &col1)
    {
        _BGColor0 = col0;
        _BGColor1 = col1;
    }

    float imWidth() { return m_GLData->glImage()._m_image->width();  }
    float imHeight(){ return m_GLData->glImage()._m_image->height(); }

    bool  isPtInsideIm(QPointF const &pt) { return m_GLData->glImage()._m_image->isPtInside(pt); }

    GLint vpWidth() { return _matrixManager.vpWidth();  }
    GLint vpHeight(){ return _matrixManager.vpHeight(); }

    cPolygon* polygon(int id);
    cPolygon* polygon();

    void setCursorShape(QPointF pos, QPointF mPos);

    void addGlPoint(QPointF pt, cOneSaisie *aSom, QPointF pt1, QPointF pt2, bool highlight);

    void setTranslation(Pt3dr trans);

    ContextMenu *contextMenu();

    void setParams(cParameters *aParams);

    void movePointWithArrows(QKeyEvent* event);

    void overlay();

    bool isFirstAction() { return m_bFirstAction; }

    int  getWindowMeanValue(QPoint pos, int r = 7); //pos = image position, r = half size of window

public slots:

    void centerViewportOnImagePosition(QPointF pt, float zoom = -1);

    void lineThicknessChanged(float);
    void gammaChanged(float);
    void pointDiameterChanged(float);
    void selectionRadiusChanged(int);
    void shiftStepChanged(float);
    void forceGray(bool);
    void showMasks(bool);

    //! Sets current zoom
    void setZoom(float val);

    void selectPoint(QString namePt);

    void setCenterType(int);

signals:

    //! Signal emitted when files are dropped on the window
    void filesDropped(const QStringList& filenames, bool setGLData = true);

    void newImagePosition(QPointF pt);

    void overWidget(void* widget);

    void gammaChangedSgnl(float gamma);

    void addPoint(QPointF point);

    void movePoint(int idPt);

    void selectPoint(int idPt);

    void removePoint(int state, int idPt);

    void newRectanglePosition(QVector <QPointF> points);

    void maskEdited();

protected:
    //! inherited from QGLWidget
    void resizeGL(int w, int h);
    void paintGL();

    //! inherited from QWidget
    void mouseDoubleClickEvent  (QMouseEvent *event);
    void mousePressEvent        (QMouseEvent *event);
    void mouseReleaseEvent      (QMouseEvent *event);
    void mouseMoveEvent         (QMouseEvent *event);

    void keyPressEvent  (QKeyEvent *event);
    void keyReleaseEvent(QKeyEvent *event);

    void wheelEvent(QWheelEvent *event);

    void dragEnterEvent(QDragEnterEvent *event);
    void dropEvent(QDropEvent *event);

    void contextMenuEvent(QContextMenuEvent *event);

    void enterEvent(QEvent *event);

    //! Current interaction mode (with mouse)
    int  m_interactionMode;

    bool m_bFirstAction;

    //! Data to display
    cGLData    *m_GLData;

    //! states if display is 2D or 3D
    bool        m_bDisplayMode2D;

    QPointF     m_lastMoveImage;
    QPoint      m_lastClickZoom;

    QPointF     m_lastPosImage;
    QPoint      m_lastPosWindow;

    bool        imageLoaded();

private:

    //! Window parameters (zoom, etc.)
    ViewportParameters _vp_Params;

    void        computeFPS(MessageToDisplay &dynMess);

    int         _frameCount;

    int         _previousTime;

    int         _currentTime;

    QTime       _time;

    MatrixManager   _matrixManager;

    cMessages2DGL   _messageManager;

    HistoryManager  _historyManager;

    ContextMenu     _contextMenu;

    int             _widgetId;

    QColor      _BGColor0;

    QColor      _BGColor1;

    cParameters* _params;

   // QPainter*   _painter;
};

#endif  /* _GLWIDGET_H */

