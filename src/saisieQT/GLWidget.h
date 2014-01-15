#ifndef _GLWIDGET_H
#define _GLWIDGET_H

#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>

#include <QtOpenGL/QGLWidget>
#include <QGLContext>
#include <QDebug>

#include "GL/glu.h"

#include <QUrl>
#include <QtGui/QMouseEvent>
#include <QSettings>
#include <QMimeData>
#include <QTime>
#include <QPainter>
//#include <QMenu> //pour contextMenuEvent

#include "Data.h"
#include "Engine.h"
#include "3DTools.h"
#include "3DObject.h"
#include "MatrixManager.h"
#include "GLWidgetSet.h"

class GLWidgetSet;



class GLWidget : public QGLWidget
{
    Q_OBJECT

public:

    //! Default constructor
    GLWidget(int idx, GLWidgetSet *theSet, const QGLWidget *shared);

    //! Destructor
    ~GLWidget(){}

    //! States if data (cloud, camera or image) is loaded
    bool hasDataLoaded(){ return (m_GLData != NULL);}

    //! Sets camera to a predefined view (top, bottom, etc.)
    void setView(VIEW_ORIENTATION orientation);

    //! Sets current zoom
    void setZoom(float value);

    //! Get current zoom
    float getZoom(){return getParams()->m_zoom;}

    void zoomFit();

    void zoomFactor(int percent);

    //! Switch between move mode and selection mode (only in 3D)
    void setInteractionMode(int mode, bool showmessage);

    bool getInteractionMode(){return m_interactionMode;}

    void setOption(QFlags<cGLData::Option> option, bool show = true);

    //! Apply selection to data
    void Select(int mode, bool saveInfos = true);

    //! Delete current polyline
    void clearPolyline()
    {
        if (hasDataLoaded())
            m_GLData->m_polygon.clear();
    }

    //! Get the selection infos stack
    QVector <selectInfos> getSelectInfos(){return _infos;}

    void applyInfos(QVector <selectInfos> &infos);

    //! Avoid all past actions
    void reset();

    //! Reset view
    void resetView(bool zoomfit = true, bool showMessage = true, bool resetMatrix = true);

    ViewportParameters* getParams(){return &_params;}

    void setGLData(cGLData* aData, bool showMessage = true, bool doZoom = true);
    cGLData* getGLData(){return m_GLData;}

    void setBackgroundColors(QColor const &col0, QColor const &col1)
    {
        _BGColor0 = col0;
        _BGColor1 = col1;
    }

    float imWidth(){  return m_GLData->glMaskedImage._m_image->width(); }
    float imHeight(){ return m_GLData->glMaskedImage._m_image->height();}

    GLint vpWidth(){  return _matrixManager.vpWidth(); }
    GLint vpHeight(){  return _matrixManager.vpHeight(); }

    cPolygon & polygon(){ return m_GLData->m_polygon;}

    void refreshMessagePosition(QPointF pos);
public slots:

    void onWheelEvent(float wheelDelta_deg);

signals:

    //! Signal emitted when files are dropped on the window
    void filesDropped(const QStringList& filenames);

protected:
    //! inherited from QGLWidget
    void resizeGL(int w, int h);
    void paintGL();

    //! inherited from QWidget
    void mouseDoubleClickEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void keyReleaseEvent(QKeyEvent *event);
    void wheelEvent(QWheelEvent* event);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void dragEnterEvent(QDragEnterEvent* event);
    void dropEvent(QDropEvent* event);

    //void contextMenuEvent(QContextMenuEvent *event);

    void Overlay();

    //! Current interaction mode (with mouse)
    int m_interactionMode;

    bool m_bFirstAction;

    //! Data to display
    cGLData    *m_GLData;

    //! states if display is 2D or 3D
    bool        m_bDisplayMode2D;

    QPointF     m_lastMoveImage;
    QPoint      m_lastClickZoom;

    QPointF     m_lastPosImage;
    QPoint      m_lastPosWindow;

private:

    //! Window parameters (zoom, etc.)
    ViewportParameters _params;

    //! selection infos stack
    QVector <selectInfos> _infos;

    void        computeFPS(MessageToDisplay &dynMess);

    int         _frameCount;
    int         _previousTime;
    int         _currentTime;    

    QTime       _time;

    MatrixManager _matrixManager;
    cMessages2DGL _messageManager;

    int         _widgetId;

    GLWidgetSet* _parentSet;

    QColor      _BGColor0;
    QColor      _BGColor1;

    QPainter*    _painter;
};

#endif  /* _GLWIDGET_H */

