#ifndef _GLWIDGET_H
#define _GLWIDGET_H

#include <QtOpenGL/QGLWidget>
#include <QGLContext>
#include <QUrl>

#include "Cloud.h"
#include "util.h"
#include "mmVector3.h"

//! Model view matrix size (OpenGL)
static const unsigned OPENGL_MATRIX_SIZE = 16;

using namespace std;

class ViewportParameters
{
public:
    //! Default constructor
    ViewportParameters();

    //! Copy constructor
    ViewportParameters(const ViewportParameters& params);

    //! Current pixel size (in 'current unit'/pixel)
    /** This scale is valid eveywhere in ortho. mode
        or at the focal distance in perspective mode.
        Warning: doesn't take current zoom into account!
    **/
    float pixelSize;

    //! Current zoom
    float zoom;

    //! Point size
    float defaultPointSize;
    //! Line width
    float defaultLineWidth;
};

class GLWidget : public QGLWidget
{
private:
    QVector <Cloud_::Cloud> m_ply;

    Q_OBJECT // must include this if you use Qt signals/slots

    QPoint              lastPos;

public:
    GLWidget(QWidget *parent = NULL);

    void addPly( const QString & );

    //! Interaction mode (with the mouse!)
    enum INTERACTION_MODE { TRANSFORM_CAMERA,
                            TRANSFORM_ENTITY,
                            SEGMENT_POINTS
    };

    //! Default message positions on screen
    enum MessagePosition {  LOWER_LEFT_MESSAGE,
                            UPPER_CENTER_MESSAGE,
                            SCREEN_CENTER_MESSAGE
    };

    //! Message type
    enum MessageType {  CUSTOM_MESSAGE,                       
                        MANUAL_SEGMENTATION_MESSAGE
    };

    //! Displays a status message in the bottom-left corner
    /** WARNING: currently, 'append' is not supported for SCREEN_CENTER_MESSAGE
        \param message message (if message is empty and append is 'false', all messages will be cleared)
        \param pos message position on screen
        \param type message type (if not custom, only one message of this type at a time is accepted)
    **/
    virtual void displayNewMessage(const QString& message,
                                   MessagePosition pos,
                                   MessageType type=CUSTOM_MESSAGE);

    //! States if a cloud is loaded
    bool hasCloudLoaded(){return m_bCloudLoaded;}

    void setCloudLoaded(bool isLoaded) { m_bCloudLoaded = isLoaded; }

    //! Sets camera to a predefined view (top, bottom, etc.)
    void setView(MM_VIEW_ORIENTATION orientation);

    //! Updates current zoom
    void updateZoom(float zoomFactor);

    //! Sets current zoom
    void setZoom(float value);

    void setInteractionMode(INTERACTION_MODE mode);

    void segment(bool inside);

    void clearPolyline();
    void closePolyline();

    void undoAll();

    GLdouble m_minX, m_maxX, m_minY, m_maxY, m_minZ, m_maxZ, m_cX, m_cY, m_cZ, m_diam;

public slots:
    void zoom();

    //called when recieving mouse wheel is rotated
    void onWheelEvent(float wheelDelta_deg);

signals:

    //! Signal emitted when files are dropped on the window
    void filesDropped(const QStringList& filenames);

    //! Signal emitted when the mouse wheel is rotated
    void mouseWheelRotated(float wheelDelta_deg);

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void mouseReleaseEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void wheelEvent(QWheelEvent* event);

    //! Initialization state
    bool m_bInitialized;

    //inherited from QWidget (drag & drop support)
    virtual void dragEnterEvent(QDragEnterEvent* event);
    virtual void dropEvent(QDropEvent* event);

    //void getContext(glDrawContext& context);

    void draw3D();

    void drawGradientBackground();

    void setStandardOrthoCenter();

    //! GL context width
    int m_glWidth;
    //! GL context height
    int m_glHeight;

    //! Returns current font size
    virtual int getFontPointSize() const;
    //! Sets current font size
    virtual void setFontPointSize(int pixelSize);

    //! Default font
    QFont m_font;

    //! States if a cloud is already loaded
    bool m_bCloudLoaded;

    //! States if selection polyline is closed
    bool m_bPolyIsClosed;

    //! Current interaction mode (with mouse)
    INTERACTION_MODE m_interactionMode;

    //! Temporary Message to display in the lower-left corner
    struct MessageToDisplay
    {
        //! Message
        QString message;
        //! Message end time (sec)
        int messageValidity_sec;
        //! Message position on screen
        MessagePosition position;
        //! Message type
        MessageType type;
    };

    //! List of messages to display
    list<MessageToDisplay> m_messagesToDisplay;

    //! Point list for polygonal selection
    QVector < QPoint > m_polygon;

    //! Viewport parameters (zoom, etc.)
    ViewportParameters m_params;
};




#endif  /* _GLWIDGET_H */

