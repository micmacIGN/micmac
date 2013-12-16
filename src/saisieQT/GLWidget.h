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


#include "Data.h"
#include "Engine.h"
#include "3DTools.h"


#include "3DObject.h"

class GLWidgetSet;

//! View orientation
enum VIEW_ORIENTATION {  TOP_VIEW,      /**< Top view (eye: +Z) **/
                         BOTTOM_VIEW,	/**< Bottom view **/
                         FRONT_VIEW,	/**< Front view **/
                         BACK_VIEW,     /**< Back view **/
                         LEFT_VIEW,     /**< Left view **/
                         RIGHT_VIEW     /**< Right view **/
};


//! Default message positions on screen
enum MessagePosition {  LOWER_LEFT_MESSAGE,
                        LOWER_RIGHT_MESSAGE,
                        LOWER_CENTER_MESSAGE,
                        UPPER_CENTER_MESSAGE,
                        SCREEN_CENTER_MESSAGE
};

//! Temporary Message to display
struct MessageToDisplay
{
    MessageToDisplay():
        color(Qt::white)
    {}

    //! Message
    QString message;

    //! color
    QColor color;

    //! Message position on screen
    MessagePosition position;
};

class GLWidget : public QGLWidget
{
    Q_OBJECT

public:

    //! Default constructor
    GLWidget(int idx, GLWidgetSet *theSet, const QGLWidget *shared);

    //! Destructor
    ~GLWidget();

//    bool eventFilter(QObject* object, QEvent* event);

    //! Interaction mode (only in 3D)
    enum INTERACTION_MODE { TRANSFORM_CAMERA,
                            SELECTION
    };



    //! Displays a status message
    /** \param message message (if message is empty, all messages will be cleared)
        \param pos message position on screen
    **/
    virtual void displayNewMessage(const QString& message,
                                   MessagePosition pos = SCREEN_CENTER_MESSAGE);

    void updateAfterSetData();
    void updateAfterSetData(bool doZoom);

    //! States if data (cloud, camera or image) is loaded
    bool hasDataLoaded();

    //! Sets camera to a predefined view (top, bottom, etc.)
    void setView(VIEW_ORIENTATION orientation);

    //! Sets current zoom
    void setZoom(float value);

    //! Get current zoom
    float getZoom(){return getParams()->m_zoom;}

    void zoomFit();

    void zoomFactor(int percent);

    //! Switch between move mode and selection mode (only in 3D)
    void setInteractionMode(INTERACTION_MODE mode, bool showmessage);

    //! Shows axis or not
    void showAxis(bool show);

    //! Shows ball or not
    void showBall(bool show);

    //! Shows cams or not
    void showCams(bool show);

    //! Shows help messages or not
    void ConstructListMessages(bool show);

    //! Shows bounding box or not
    void showBBox(bool show);

    //! States if help messages should be displayed
    bool showMessages(){return m_bDrawMessages;}

    //! Display help messages for selection mode
    void displaySelectionMessages();

    //! Display help messages for move mode
    void displayMoveMessages();

    //! Select points with polyline
    void Select(int mode);

    void Select(int mode, bool saveInfos);

    //! Delete current polyline
    void clearPolyline();

    //!Undo last action
    void undo();

     //! Undo all past selection actions
    void undoAll();

    void getProjection(QPointF &P2D, Pt3dr P);

    QVector <selectInfos> getSelectInfos(){return m_infos;}

    //! Avoid all past actions
    void reset();

    //! Reset view
    void resetView();

    void resetRotationMatrix();

    void resetTranslationMatrix();

    ViewportParameters* getParams(){return &m_params;}

//    void enableOptionLine();
//    void disableOptionLine();

    void setGLData(cGLData* aData);
    cGLData* getGLData(){return m_GLData;}

    void setBackgroundColors(QColor const &col0, QColor const &col1);

    cPolygon PolyImageToWindow(cPolygon polygon);

    int renderLineText(MessageToDisplay messageTD, int x, int y, int sizeFont = 10);

public slots:
    void zoom();

    //! called when receiving mouse wheel is rotated
    void onWheelEvent(float wheelDelta_deg);

signals:

    //! Signal emitted when files are dropped on the window
    void filesDropped(const QStringList& filenames);

    void selectedPoint(uint idCloud, uint idVertex,bool selected);

protected:
    void resizeGL(int w, int h);
    void paintGL();
    void mouseDoubleClickEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void keyReleaseEvent(QKeyEvent *event);
    void wheelEvent(QWheelEvent* event);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);

    //inherited from QWidget (drag & drop support)
    virtual void dragEnterEvent(QDragEnterEvent* event);
    virtual void dropEvent(QDropEvent* event);

    //! Draw widget gradient background
    void drawGradientBackground();
    
    //! Draw selection polygon
    void drawPolygon();

    QPointF WindowToImage(const QPointF &pt);

    QPointF ImageToWindow(const QPointF &im);

    //! GL context aspect ratio (width/height)
    float m_glRatio;

    //! Default font
    QFont m_font;

    //! States if messages should be displayed
    bool m_bDrawMessages;

    //! Current interaction mode (with mouse)
    INTERACTION_MODE m_interactionMode;

    bool m_bFirstAction;

    //! List of messages to display
    list<MessageToDisplay> m_messagesToDisplay;

    //QString     m_messageFPS;

    //! Viewport parameters (zoom, etc.)
    ViewportParameters m_params;

    //! Data to display
    cGLData    *m_GLData;

    //! selection infos stack
    QVector <selectInfos> m_infos;

    //! states if display is 2D or 3D
    bool        m_bDisplayMode2D;

    //! data position in the gl viewport
    GLfloat     m_glPosition[2];

    QPointF     m_lastMoveImage;
    QPoint      m_lastClickZoom;

    QPointF     m_lastPosImage;
    QPoint      m_lastPosWindow;

private:

    void        setProjectionMatrix();
    void        computeFPS(MessageToDisplay &dynMess);

    int         _frameCount;
    int         _previousTime;
    int         _currentTime;    

    GLfloat     _g_tmpoMatrix[9];
    GLfloat     _g_rotationOx[9];
    GLfloat     _g_rotationOy[9];
    GLfloat     _g_rotationOz[9];
    GLfloat     _g_rotationMatrix[9];
    GLfloat     _g_glMatrix[16];
    QTime       _time;

    GLdouble    *_mvmatrix;
    GLdouble    *_projmatrix;
    GLint       *_glViewport;

    //bool        _bDataLoaded;
    int         _idx;

    GLWidgetSet* _parentSet;

    QColor      _BGColor0;
    QColor      _BGColor1;
};

#endif  /* _GLWIDGET_H */

