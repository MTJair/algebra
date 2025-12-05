<?php
session_start();
if (!isset($_SESSION['name']) || $_SESSION['module'] !== 'algebra') {
    header('Location: http://www.quantcodehub.com/clases.php');
    exit();}
?>
<!DOCTYPE html>
<html>
<head lang="en">
    <meta charset="UTF-8">

    <!--Page Title-->
    <title>Math!</title>

    <!--Meta Keywords and Description-->
    <meta name="keywords" content="">
    <meta name="description" content="">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no"/>

    <!--Favicon-->
    <link rel="shortcut icon" href="images/math.png" title="Favicon"/>

    <!-- Main CSS Files -->
    <link rel="stylesheet" href="css/style.css">

    <!-- Namari Color CSS -->
    <link rel="stylesheet" href="css/namari-color.css">

    <!--Icon Fonts - Font Awesome Icons-->
    <link rel="stylesheet" href="css/font-awesome.min.css">

    <!-- Animate CSS-->
    <link href="css/animate.css" rel="stylesheet" type="text/css">

    <!--Google Webfonts-->
    <link href='https://fonts.googleapis.com/css?family=Open+Sans:400,300,600,700,800' rel='stylesheet' type='text/css'>

    <style type="text/css">
    .text-center {
        text-align: center;}
    .container {    
    display: flex; !important;
    justify-content: center; !important;}
    .col-2.icon-block {
      border: 2px solid black; /* Establece un borde sólido de 2px en color negro */
      border-radius: 5px; /* Agrega esquinas redondeadas */
      padding: 10px; /* Añade un relleno interno de 10px */
      cursor: pointer; /* Cambia el cursor al pasar sobre el elemento */
      transition: all 0.3s ease; /* Agrega una transición suave a todos los cambios */
    }
    /* Cambia el color del borde al pasar el ratón sobre el elemento */
    .col-2.icon-block:hover {
      border-color: gray;
    }
    #justificado {text-align: justify;}
    #justificado2{            
        text-align: justify;
        margin-left: 20px; /* Margen a la izquierda */
        margin-right: 20px; /* Margen a la derecha */
    }    
    #centrado {text-align: center;}
    #derecha {text-align: right;}
    }
        .button-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
        } 
        .subtema-button {
            background-color: #82E0AA;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
            transition: background-color 0.3s ease;
        }
        .subtema-button:hover {
            background-color: #322E2D;
        }
        .subtema-button-2 {
            background-color: #636260;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
            transition: background-color 0.3s ease;
        }
        .subtema-button-2:hover {
            background-color: #322E2D;
        }
        #justificado a {
            color: blue;
            font-weight: bold;
            white-space: nowrap;  /* Evita el salto de línea */
            text-transform: none; /* Evita que el texto esté en mayúsculas */
            display: inline;      /* Asegura que el enlace se comporte como un elemento en línea */
        }
    </style>

    <script type="text/javascript" async
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

</head>
<body>

<!-- Preloader -->
<div id="preloader">
    <div id="status" class="la-ball-triangle-path">
        <div></div>
        <div></div>
        <div></div>
    </div>
</div>
<!--End of Preloader-->

<div class="page-border" data-wow-duration="0.7s" data-wow-delay="0.2s">
    <div class="top-border wow fadeInDown animated" style="visibility: visible; animation-name: fadeInDown;"></div>
    <div class="right-border wow fadeInRight animated" style="visibility: visible; animation-name: fadeInRight;"></div>
    <div class="bottom-border wow fadeInUp animated" style="visibility: visible; animation-name: fadeInUp;"></div>
    <div class="left-border wow fadeInLeft animated" style="visibility: visible; animation-name: fadeInLeft;"></div>
</div>

<div id="wrapper">

    <header id="banner" class="scrollto clearfix" data-enllax-ratio=".5">
        <div id="header" class="nav-collapse">
            <div class="row clearfix">
                <div class="col-1">

                    <!--Main Navigation-->
                    <nav id="nav-main">
                        <ul>
                            <li>
                                <a href="/clases.php">Inicio</a>
                            </li>
                            <li>
                                <a href="#clients">Álgebra</a>
                            </li>
                        </ul>
                    </nav>
                    <!--End of Main Navigation-->

                    <div id="nav-trigger"><span></span></div>
                    <nav id="nav-mobile"></nav>

                </div>
            </div>
        </div><!--End of Header-->

    </header>

    <!--Main Content Area-->
    <main id="content">

        <!--Calculadoras-->
        <section id="clients" class="scrollto clearfix">
            <div class="row clearfix">
                <div class="col-3">

                    <div class="section-heading">
                        <h3>Matemáticas</h3>
                        <h2 class="section-title">Álgebra</h2>
                        <div class="button-container">
                            <a href="" class="subtema-button">Matrices y determinantes</a>
                            <a href="/ecuacionesalgebraicas.php" class="subtema-button">Ecuaciones algebraicas</a>
                            <a href="/sistemaseclineales.php" class="subtema-button">Sistemas de ecuaciones lineales</a> 
                            <a href="/metresolsisteclineales.php" class="subtema-button">Métodos de resolución de sistemas de ecuaciones lineales</a>
                            <a href="/espaciosvectoriales.php" class="subtema-button">Espacios vectoriales</a>
                            <a href="/aplicacioneslineales.php" class="subtema-button">Aplicaciones lineales</a>
                            <a href="/pca.php" class="subtema-button">PCA: análisis de componentes principales</a>
                            <a href="/svd.php" class="subtema-button">SVD: descomposición en valores singulares</a>
                            <a href="/algebra_evaluaciones.php" class="subtema-button-2">Evaluaciones</a>
                        </div>
                    </div>

                </div>

            <div class="col-2-3">

<!-- Copia todo esto dentro de la etiqueta <body> de tu página HTML -->
<section id="tema7">
  <h1>Tema 7 — PCA: Análisis de Componentes Principales</h1>
  <p>Este tema presenta una explicación completa y didáctica sobre el Análisis de Componentes Principales (PCA): autovalores, autovectores, diagonalización, matrices simétricas, formas cuadráticas, matrices ortogonales y el eje de componentes principales. Incluye ejemplos resueltos, código en <code>Python</code> y aplicaciones a imágenes y señales.</p>

  <!-- 7.1 ----------------------------------------------------------------------------- -->
  <article id="7-1">
    <h2>7.1 Autovalores y autovectores</h2>

    <h4>Definición</h4>
    <p>Sea \(A \in \mathbb{R}^{n \times n}\). Un número \(\lambda\) es un <strong>autovalor</strong> de A si existe un vector no nulo \(v\) tal que:
      <blockquote>
        \( A v = \lambda v. \)
      </blockquote>
    El vector \(v\) se llama <strong>autovector</strong> asociado a \(\lambda\).</p>

    <h4>Forma práctica</h4>
    <p>Resolver:
      <blockquote>\( (A - \lambda I)v = 0 \)</blockquote>
    <br>
    significa que \(\lambda\) satisface:
    <br>
      <blockquote>\( \det(A - \lambda I)=0. \)</blockquote>
    </p>

    <h4>Ejemplo en Python</h4>
    <pre><code class="language-python">
import numpy as np

A = np.array([[2, 1],
              [1, 2]], float)

eigvals, eigvecs = np.linalg.eig(A)
print("Autovalores:", eigvals)
print("Autovectores:\n", eigvecs)
</code></pre>
    <br>
    <h4>Ejercicio</h4>
    <p>Verifica que cada autovector realmente cumple \(A v = \lambda v\).</p>
    <pre><code class="language-python">
for i in range(len(eigvals)):
    print("Chequeo", i, ":", A @ eigvecs[:,i], "≈", eigvals[i] * eigvecs[:,i])
</code></pre>
  </article>
  <br>
  <!-- 7.2 ----------------------------------------------------------------------------- -->
  <article id="7-2">
    <h2>7.2 Diagonalización de matrices</h2>

    <h4>Definición</h4>
    <p>Una matriz A es <strong>diagonalizable</strong> si existe una matriz invertible P tal que:
      <blockquote>\( P^{-1} A P = D, \)</blockquote>
    donde D es diagonal.</p>

    <h4>Teorema</h4>
    <p>A es diagonalizable si tiene n autovectores linealmente independientes.</p>

    <h4>Ejemplo en Python: diagonalización</h4>
    <pre><code class="language-python">
A = np.array([[4,1],
              [1,4]], float)

eigvals, P = np.linalg.eig(A)
D = np.diag(eigvals)

print("Matriz diagonal D:\n", D)
print("Verificación P D P^{-1}:\n", P @ D @ np.linalg.inv(P))
</code></pre>

    <br>
    <h4>Ejercicio</h4>
    <p>Comprueba que la reconstrucción coincide con A a 5 decimales.</p>
  </article>

  <!-- 7.3 ----------------------------------------------------------------------------- -->
  <article id="7-3">
    <h2>7.3 Matrices simétricas y formas cuadráticas</h2>

    <h4>Matrices simétricas</h4>
    <p>Una matriz A es <strong>simétrica</strong> si:
      <blockquote>\( A = A^T. \)</blockquote>
        <br>
    Las matrices de covarianza siempre son simétricas.</p>

    <h4>Teorema importante</h4>
    <p>Si A es simétrica entonces:
      <ul>
        <li>todos sus autovalores son reales,</li>
        <li>existe una base ortonormal de autovectores,</li>
        <li>puede diagonalizarse mediante una matriz ortogonal.</li>
      </ul>
    </p>

    <h4>Forma cuadrática</h4>
    <p>Dada A simétrica, la expresión:
      <blockquote>\( Q(x) = x^T A x \)</blockquote> <br>
    es una <strong>forma cuadrática</strong>.  
    PCA estudia la forma cuadrática ligada a la matriz de covarianza.</p>

    <h4>Ejemplo en Python</h4>
    <pre><code class="language-python">
A = np.array([[3,1],
              [1,2]], float)

# forma cuadrática Q(x)
def Q(x):
    return x.T @ A @ x

x = np.array([1,2], float)
print("Q(x) =", Q(x))
</code></pre>

    <br>
    <h4>Ejercicio</h4>
    <p>Calcula Q(x) para varios x y observa el crecimiento en dirección del autovector principal.</p>
  </article>

  <!-- 7.4 ----------------------------------------------------------------------------- -->
  <article id="7-4">
    <h2>7.4 Matrices ortogonales</h2>

    <h4>Definición</h4>
    <p>Una matriz P es <strong>ortogonal</strong> si:
      <blockquote>\( P^T P = I. \)</blockquote>
    En este caso \(P^{-1} = P^T\).</p>

    <h4>Interpretación geométrica</h4>
    <p>Las matrices ortogonales representan:
      <ul>
        <li>rotaciones,</li>
        <li>reflexiones,</li>
        <li>cambios de base que preservan ángulos y longitudes.</li>
      </ul>
    </p>

    <h4>Ejemplo en Python</h4>
    <pre><code class="language-python">
theta = np.pi/4
P = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

print("P^T P:\n", P.T @ P)
</code></pre>

    <br>
    <h4>Ejercicio</h4>
    <p>Genera matrices ortogonales aleatorias usando descomposición QR.</p>
  </article>

  <!-- 7.5 ----------------------------------------------------------------------------- -->
  <article id="7-5">
    <h2>7.5 Eje de componentes principales</h2>

    <h4>Idea central del PCA</h4>
    <p>El PCA encuentra la dirección donde los datos presentan la mayor variabilidad.  
    Esa dirección es el <strong>autovector asociado al mayor autovalor</strong> de la matriz de covarianza.</p>

    <h4><b>Proceso matemático</b></h4>
    <br>
    <ol>
      <li>Centra los datos: \(X_c = X - \bar{X}\).</li>    <br>
      <li>Calcula la matriz de covarianza \(C = \frac{1}{n-1} X_c^T X_c\).</li>    <br>
      <li>Obtén autovalores y autovectores de C.</li>    <br>
      <li>Ordena de mayor a menor autovalor.</li>    <br>
      <li>El primer componente es la dirección del máximo varianza.</li>
    </ol>

    <br>
    <h4>Ejemplo en Python (2D)</h4>
    <pre><code class="language-python">
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)
X = np.random.multivariate_normal([0,0], [[3,1],[1,1]], 300)

# centrar
Xc = X - X.mean(axis=0)

# covarianza
C = np.cov(Xc.T)

# autovalores y autovectores
eigvals, eigvecs = np.linalg.eig(C)

# componente principal
v1 = eigvecs[:, np.argmax(eigvals)]

# graficar datos y primer componente
plt.scatter(X[:,0], X[:,1], alpha=0.3)
plt.quiver(X.mean(axis=0)[0], X.mean(axis=0)[1],
           v1[0], v1[1], color='red', scale=3)
plt.title("Primer componente principal")
plt.grid(True)
plt.show()
</code></pre>

    <br>
    <h4>Ejercicio</h4>
    <p>Proyecta los datos sobre el primer componente principal y calcula la varianza explicada.</p>
  </article>

  <!-- 7.6 ----------------------------------------------------------------------------- -->
  <article id="7-6">
    <h2>7.6 Análisis de componentes principales para imágenes y señales</h2>

    <h4>PCA en imágenes</h4>
    <p>Una imagen se representa como una matriz.  
    PCA permite:
      <ul>
        <li>Reducir dimensionalidad</li>
        <li>Reconstruir imágenes con menos información</li>
        <li>Eliminar ruido</li>
        <li>Comprimir datos</li>
      </ul>
    </p>

    <h4>Ejemplo en Python: compresión de imagen (en escala de grises)</h4>
    <pre><code class="language-python">
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# cargar imagen
img = imread("imagen.jpg")
img = img.mean(axis=2)   # gris

# aplicar PCA usando SVD (equivalente numérico)
U, S, Vt = np.linalg.svd(img, full_matrices=False)

# reconstruir usando k componentes
k = 50
img_k = (U[:, :k] * S[:k]) @ Vt[:k, :]

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(img_k, cmap='gray')
plt.title(f"Reconstrucción con k={k}")
plt.show()
</code></pre>

    <br>
    <h4>PCA en señales</h4>
    <p>En señales (por ejemplo audio o series temporales), PCA permite:</p>
    <ul>
      <li>Eliminar ruido</li>
      <li>Detectar componentes dominantes</li>
      <li>Reducir dimensionalidad de ventanas temporales</li>
    </ul>

    <br>
    <h4>Ejemplo en Python: señal ruidosa</h4>
    <pre><code class="language-python">
import numpy as np
import matplotlib.pyplot as plt

# señal seno + ruido
t = np.linspace(0, 10, 2000)
signal = np.sin(2*np.pi*2*t)
noise = 0.3*np.random.randn(len(t))
x = signal + noise

# construir ventanas (embedding)
window = 30
X = np.array([x[i:i+window] for i in range(len(x)-window)])

# PCA en la matriz de ventanas
Xc = X - X.mean(axis=0)
C = np.cov(Xc.T)
eigvals, eigvecs = np.linalg.eig(C)

# reconstrucción usando primer componente
pc1 = eigvecs[:, np.argmax(eigvals)]
proj = Xc @ pc1

plt.plot(t[:-window], proj)
plt.title("Señal PCA (primer componente)")
plt.grid(True)
plt.show()
</code></pre>

    <br>
    <h4>Ejercicio</h4>
    <p>Experimenta con distintos tamaños de ventana y observa cómo cambia la capacidad de PCA para separar ruido.</p>
  </article>

  <footer>
    <h3>Actividades y tareas</h3>
    <ol>
      <li>Diagonaliza matrices de covarianza generadas aleatoriamente y verifica la ortogonalidad de los autovectores.</li>
      <li>Implementa PCA desde cero usando únicamente álgebra lineal (sin librerías avanzadas).</li>
      <li>Aplica PCA a un conjunto de imágenes y estudia la cantidad mínima de componentes necesarias para buena reconstrucción.</li>
      <li>Extrae patrones principales en señales de audio usando PCA.</li>
    </ol>

    <p>— <strong>Prof. Eder Díaz</strong></p>
  </footer>
</section>

            </div>

            </div>
        </section>
        <!--End of Calculadoras-->

    </main>
    <!--End Main Content Area-->

</div>

<!-- Include JavaScript resources -->
<script src="js/jquery.1.8.3.min.js"></script>
<script src="js/wow.min.js"></script>
<script src="js/featherlight.min.js"></script>
<script src="js/featherlight.gallery.min.js"></script>
<script src="js/jquery.enllax.min.js"></script>
<script src="js/jquery.scrollUp.min.js"></script>
<script src="js/jquery.easing.min.js"></script>
<script src="js/jquery.stickyNavbar.min.js"></script>
<script src="js/jquery.waypoints.min.js"></script>
<script src="js/images-loaded.min.js"></script>
<script src="js/lightbox.min.js"></script>
<script src="js/site.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://kit.fontawesome.com/30c014f357.js" crossorigin="anonymous"></script>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

</body>
</html>