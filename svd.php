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
<section id="tema8">
  <h1>Tema 8 — SVD: Descomposición en Valores Singulares</h1>
  <p>Este tema desarrolla la descomposición en valores singulares (SVD), uno de los métodos fundamentales en álgebra lineal aplicada. Se estudian los valores singulares, la interpretación geométrica de la SVD, su cálculo numérico y sus aplicaciones en reducción de datos y reducción de variables. Cada sección incluye ejemplos y código en <code>Python</code>.</p>

  <!-- 8.1 ----------------------------------------------------------------------------- -->
  <article id="8-1">
    <h2>8.1 Valores singulares</h2>

    <h4>Definición</h4>
    <p>Sea A una matriz real de dimensión \(m \times n\). Los <strong>valores singulares</strong> de A son los números:</p>

    <blockquote>
      \(\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0\)
    </blockquote>

    <p>donde \(\sigma_i\) son las raíces cuadradas de los autovalores de la matriz simétrica:</p>

    <blockquote>
      \(A^T A\).
    </blockquote>

    <p>La cantidad r es el <strong>rango</strong> de la matriz A.</p>

    <br>
    <h4>Propiedades importantes</h4>
    <br>
    <ul>
      <li>Todos los valores singulares son reales y no negativos.</li>
      <li>Coinciden con las longitudes de los ejes principales de la transformación lineal inducida por A.</li>
      <li>Se relacionan con el número de condición: <br> <em>cond(A) = σ₁ / σᵣ</em>.</li>
    </ul>
    <br>
    <h4>Ejemplo en Python</h4>
    <pre><code class="language-python">
import numpy as np

A = np.array([[3, 1],
              [1, 3],
              [0, 2]], float)

# valores singulares
U, S, Vt = np.linalg.svd(A)

print("Valores singulares:", S)
</code></pre>
    <br>
    <h4>Ejercicio</h4>
    <p>Calcula los autovalores de \(A^T A\) y confirma que \(\sigma_i = \sqrt{\lambda_i}\).</p>

  </article>

  <!-- 8.2 ----------------------------------------------------------------------------- -->
  <article id="8-2">
    <h2>8.2 Descomposición SVD: interpretación geométrica</h2>

    <h4>Definición</h4>
    <p>La SVD descompone cualquier matriz rectangular \(A\in\mathbb{R}^{m\times n}\) como:</p>

    <blockquote>
      \(A = U \, \Sigma \, V^T\)
    </blockquote>

    donde:
    <ul>
      <li><strong>U</strong> es una matriz ortogonal \(m \times m\),</li>
      <li><strong>Σ</strong> es una matriz diagonal rectangular con valores singulares,</li>
      <li><strong>V</strong> es ortogonal \(n \times n\).</li>
    </ul>
    <br>
    <h4>Interpretación geométrica</h4>
    <p>La transformación \(x \mapsto Ax\) puede verse como:</p>
    <ol>
      <li>Aplicar una rotación (o reflexión) dada por \(V^T\).</li>
      <li>Estirar o comprimir según los valores singulares en Σ.</li>
      <li>Aplicar otra rotación dada por U.</li>
    </ol>

    <p>Así, SVD separa la transformación en rotación → estiramiento → rotación.</p>

    <h4>Ejemplo en Python: visualizando el efecto geométrico</h4>
    <pre><code class="language-python">
import numpy as np
import matplotlib.pyplot as plt

# matriz
A = np.array([[2,1],
              [1,3]], float)

# SVD
U, S, Vt = np.linalg.svd(A)

# círculo unitario
theta = np.linspace(0, 2*np.pi, 200)
circle = np.vstack([np.cos(theta), np.sin(theta)])

# aplicar A
ellipse = A @ circle

plt.figure(figsize=(6,6))
plt.plot(circle[0], circle[1], label='Círculo unitario')
plt.plot(ellipse[0], ellipse[1], label='Imagen bajo A')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('Transformación geométrica de A')
plt.show()
</code></pre>
    <br>
    <h4>Ejercicio</h4>
    <p>Modifica la matriz A y observa cómo cambian el estiramiento (valores singulares) y la orientación del elipsoide.</p>

  </article>

  <!-- 8.3 ----------------------------------------------------------------------------- -->
  <article id="8-3">
    <h2>8.3 Cálculo de la SVD</h2>

    <h4>Idea numérica</h4>
    <p>La SVD se calcula encontrando autovectores de las matrices simétricas:</p>

    <ul>
      <li>\(V\) proviene de los autovectores de \(A^T A\).</li>    <br>
      <li>\(U\) proviene de los autovectores de \(A A^T\).</li>    <br>
      <li>Los valores singulares son \(\sigma_i = \sqrt{\lambda_i}\).</li>
    </ul>

    <p>Numéricamente se usa una variante del método QR que garantiza estabilidad.</p>

    <h4>Cálculo manual en un ejemplo 2×2</h4>
    <pre><code class="language-python">
import numpy as np

A = np.array([[4, 0],
              [3, -5]], float)

# 1) autovalores de A^T A
ATA = A.T @ A
eigvals, V = np.linalg.eig(ATA)

# valores singulares
S = np.sqrt(eigvals)

# U = A V / sigma
U = (A @ V) / S

print("A:\n", A)
print("\nValores singulares:", S)
print("\nV (autovectores de A^T A):\n", V)
print("\nU calculada manualmente:\n", U)
</code></pre>
    <br>
    <h4>Ejercicio</h4>
    <p>Verifica que \(A = U \Sigma V^T\) reconstruye la matriz original A.</p>

  </article>

  <!-- 8.4 ----------------------------------------------------------------------------- -->
  <article id="8-4">
    <h2>8.4 Aplicaciones: Reducción de Datos y Reducción de Variables</h2>

    <h4><b>8.4.1 Reducción de Datos (Data Reduction)</b></h4>
    <p>La SVD permite aproximar una matriz A por una de rango k &lt; r conservando la mayor parte de la información:</p>

    <blockquote>
      \(A_k = U_k \Sigma_k V_k^T\)
    </blockquote>

    <p>Esta aproximación minimiza el error cuadrático (<em>teorema de Eckart-Young</em>). Se usa en:</p>
    <ul>
      <li>Compresión de imágenes</li>
      <li>Compresión de audio</li>
      <li>Filtrado de ruido</li>
      <li>Análisis de preferencias de usuarios (recomendadores)</li>
    </ul>
    <br>
    <h4>Ejemplo en Python: compresión de imagen</h4>
    <pre><code class="language-python">
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# cargar imagen
img = imread("imagen.jpg")
img_gray = img.mean(axis=2)

U, S, Vt = np.linalg.svd(img_gray, full_matrices=False)

k = 50  # número de valores singulares
img_k = (U[:, :k] * S[:k]) @ Vt[:k, :]

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(img_gray, cmap='gray')
plt.title("Imagen original (gris)")

plt.subplot(1,2,2)
plt.imshow(img_k, cmap='gray')
plt.title("Reconstrucción con k=50")
plt.show()
</code></pre>
    <br>
    <h4>Ejercicio</h4>
    <p>Varía el valor de k y calcula el error de reconstrucción con la norma Frobenius: <br> <em>error = ||A − A_k||_F</em>.</p>

    <h4><b>8.4.2 Reducción de Variables (Variable Reduction)</b></h4>
    <p>SVD es fundamental en modelos de alta dimensión. Las columnas de V (componentes principales) permiten seleccionar nuevas variables ortogonales que capturan la mayor variabilidad.</p>

    <p>Esto es equivalente al PCA cuando los datos están centrados.</p>

    <h4>Ejemplo en Python: reducción a k variables</h4>
    <pre><code class="language-python">
import numpy as np

# datos sintéticos de 100 muestras y 10 variables
np.random.seed(1)
X = np.random.randn(100, 10)

# centrar
Xc = X - X.mean(axis=0)

# SVD
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

# proyectar sobre los primeros k componentes
k = 3
X_k = Xc @ Vt[:k].T

print("Dimensiones originales:", X.shape)
print("Nuevas dimensiones reducidas:", X_k.shape)
</code></pre>
    <br>
    <h4>Ejercicio</h4>
    <p>Calcula la varianza explicada por los primeros k valores singulares: <br>
      <em>var_expl = (σ₁² + … + σ_k²) / (σ₁² + … + σ_r²)</em>.
    </p>

  </article>

  <footer>
    <h3>Actividades y tareas</h3>
    <ol>
      <li>Implementa la SVD manualmente para una matriz 3×3 usando autovectores de \(A^T A\) y \(A A^T\).</li>
      <li>Simula matrices de diferente condición (σ₁/σᵣ grandes) y analiza la estabilidad numérica.</li>
      <li>Aplica SVD a datos de precios financieros para encontrar factores dominantes.</li>
      <li>Realiza compresión de imágenes variando k y estudia la curva de error vs. k.</li>
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