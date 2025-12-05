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
<section id="tema5">
  <h1>Tema 5 — Espacios vectoriales</h1>
  <p>Este tema introduce los conceptos fundamentales de los espacios vectoriales en álgebra lineal: combinaciones lineales, independencia, subespacios, bases y dimensión.  
  Cada subtema incluye ejemplos, intuiciones geométricas y código en <code>Python</code> (listo para Spyder) para experimentar numéricamente.</p>

  <!-- 5.1 ----------------------------------------------------------------------------- -->
  <article id="5-1">
    <h2>5.1 Combinaciones lineales</h2>
    <h4>Definición</h4>
    <p>Dados vectores \(v_1, v_2, \dots, v_k \in \mathbb{R}^n\), una <strong>combinación lineal</strong> es cualquier vector de la forma:
      <br>
      <em>\( \alpha_1 v_1 + \alpha_2 v_2 + \dots + \alpha_k v_k \)</em>,
      <br>
      donde \(\alpha_i \in \mathbb{R}\).
    </p>

    <h4>Interpretación geométrica</h4>
    <p>Las combinaciones lineales generan todos los puntos alcanzables “estirando” y “sumando” los vectores iniciales.</p>

    <h4>Ejemplo simple</h4>
    <br>
    <blockquote>
      Sean \( v_1=(1,2) \) y \( v_2=(2,-1) \).  
      <br>Cualquier vector del plano puede escribirse como combinación lineal de ellos porque forman una base de \(\mathbb{R}^2\).
    </blockquote>
    <br>
    <h4>Ejemplo en Python</h4>
    <pre><code class="language-python">
import numpy as np

# vectores base
v1 = np.array([1, 2])
v2 = np.array([2, -1])

# coeficientes
a, b = 3, -2

combo = a * v1 + b * v2
print("Combinación lineal:", combo)
</code></pre>
    <br>
    <h4>Ejercicio en Python</h4>
    <p>Escribe una función que, dados vectores columna, determine si un vector objetivo es combinación lineal resolviendo Ax=b.</p>
    <pre><code class="language-python">
def es_combinacion_lineal(vectores, objetivo):
    A = np.column_stack(vectores)
    objetivo = np.array(objetivo)
    x, *_ = np.linalg.lstsq(A, objetivo, rcond=None)
    return A @ x, x  # aproximación y coeficientes

v1 = np.array([1, 1, 0])
v2 = np.array([0, 1, 1])
v3 = np.array([1, 0, 1])
objetivo = np.array([2, 2, 1])

aprox, coef = es_combinacion_lineal([v1, v2, v3], objetivo)
print("Vector generado:", aprox)
print("Coeficientes:", coef)
</code></pre>
  </article>
    <br>
  <!-- 5.2 ----------------------------------------------------------------------------- -->
  <article id="5-2">
    <h2>5.2 Independencia lineal</h2>

    <h4>Definición</h4>
    <p>Los vectores \(v_1,\dots,v_k\) son <strong>linealmente independientes</strong> si:
      <blockquote>\( \alpha_1 v_1+\dots+\alpha_k v_k=0 \Rightarrow \alpha_1=\dots=\alpha_k=0. \)</blockquote>
    </p>

    <h4>Criterios prácticos</h4>
    <br>
    <ul>
      <li>Los vectores son independientes ⇔ rank(A) = número de vectores.</li>
      <li>Si uno puede escribirse como combinación de los otros ⇒ dependencia.</li>
    </ul>
    <br>
    <h4>Ejemplo en Python: verificar independencia</h4>
    <pre><code class="language-python">
import numpy as np

def son_independientes(vectores):
    A = np.column_stack(vectores)
    rango = np.linalg.matrix_rank(A)
    return rango == len(vectores)

v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])  # múltiplo → dependiente
v3 = np.array([1, 0, 1])

print("Independencia:", son_independientes([v1, v2, v3]))
</code></pre>
    <br>
    <h4>Ejercicio en Python</h4>
    <p>Encuentra una combinación lineal que demuestre dependencia cuando los vectores no son independientes.</p>
    <pre><code class="language-python">
A = np.column_stack([v1, v2, v3])
U, S, Vt = np.linalg.svd(A)

# vectores con singular values ~ 0 → combinación de dependencia
nullspace = Vt[-1]
print("Combinación que da 0:", nullspace)
print("Verificación:", A @ nullspace)
</code></pre>
  </article>
    <br>
  <!-- 5.3 ----------------------------------------------------------------------------- -->
  <article id="5-3">
    <h2>5.3 Subespacios. Bases</h2>

    <h4>Subespacio vectorial</h4>
    <p>Un conjunto \(W \subseteq \mathbb{R}^n\) es un <strong>subespacio</strong> si:
      <ul>
        <li>contiene al vector cero,</li>
        <li>es cerrado bajo suma,</li>
        <li>es cerrado bajo multiplicación por escalares.</li>
      </ul>
    </p>

    <h4>Bases</h4>
    <p>Una base de un espacio vectorial es un conjunto de vectores linealmente independientes que generan todo el espacio.</p>

    <h4>Ejemplo en Python: encontrar una base del subespacio columna</h4>
    <pre><code class="language-python">
import numpy as np

A = np.array([[1,2,3],
              [2,4,6],
              [1,0,1]], float)

# Usamos SVD para obtener base del espacio columna
U, S, Vt = np.linalg.svd(A)
rango = np.sum(S > 1e-10)
base_columna = U[:, :rango]

print("Rango:", rango)
print("Base del espacio columna:")
print(base_columna)
</code></pre>

    <h4>Ejercicio: base del espacio nulo</h4>
    <pre><code class="language-python">
def nullspace(A, tol=1e-12):
    U, S, Vt = np.linalg.svd(A)
    r = np.sum(S > tol)
    return Vt[r:].T

ns = nullspace(A)
print("Base del espacio nulo:")
print(ns)
</code></pre>
  </article>
    <br>
  <!-- 5.4 ----------------------------------------------------------------------------- -->
  <article id="5-4">
    <h2>5.4 Matrices y cambios de base</h2>

    <h4>Idea fundamental</h4>
    <p>Un mismo vector puede expresarse en diferentes bases.  
    Si \(B=\{b_1,\dots,b_n\}\) y \(C=\{c_1,\dots,c_n\}\) son bases, entonces las coordenadas del vector cambian mediante una matriz de cambio de base.</p>

    <h4>Definición</h4>
    <p>La matriz que cambia coordenadas de la base B a la base C se define como:
      <blockquote>\( P_{C \leftarrow B} = [\,c_1\ c_2\ \dots\ c_n\,]^{-1}[\,b_1\ b_2\ \dots\ b_n\,]. \)</blockquote>
    </p>

    <h4>Ejemplo en Python: cambiar coordenadas</h4>
    <pre><code class="language-python">
import numpy as np

# base estándar
e1 = np.array([1,0])
e2 = np.array([0,1])

# nueva base
b1 = np.array([2,1])
b2 = np.array([1,1])

B = np.column_stack([b1, b2])
E = np.eye(2)  # base estándar

# matriz de cambio de base: de B a estándar
P = np.linalg.inv(B) @ E
print("Cambio de base B -> estándar:")
print(P)

# coordenadas de un vector en la base B
v = np.array([3,4])
coords_B = np.linalg.inv(B) @ v
print("Coordenadas de v en base B:", coords_B)
</code></pre>
    <br>
    <h4>Ejercicio en Python</h4>
    <p>Comprueba que cambiar de base ida y vuelta recupera el vector original.</p>
    <pre><code class="language-python">
# cambio estándar -> base B
P_inv = B @ np.linalg.inv(E)

recuperado = B @ coords_B
print("Vector después de ida y vuelta:", recuperado)
</code></pre>
  </article>
    <br>
  <!-- 5.5 ----------------------------------------------------------------------------- -->
  <article id="5-5">
    <h2>5.5 Dimensión</h2>

    <h4>Definición</h4>
    <p>La dimensión de un espacio vectorial es el número de vectores en cualquier base del espacio.  
    Por ejemplo:
      <ul>
        <li>dim(\(\mathbb{R}^n\)) = n,</li>
        <li>dim(Espacio columna de A) = rank(A),</li>
        <li>dim(Espacio nulo de A) = n − rank(A).</li>
      </ul>
    </p>

    <h4>Teorema fundamental</h4>
    <br>
    <blockquote>
      Para una matriz \(A\in\mathbb{R}^{m\times n}\):
      <br>
      <strong>rango(A) + dimensión(nulidad(A)) = n.</strong>
    </blockquote>
    <br>
    <h4>Cálculo en Python: rango, nulidad y dimensión</h4>
    <pre><code class="language-python">
import numpy as np

A = np.array([[1,2,3],
              [2,4,6],
              [1,0,1]], float)

rango = np.linalg.matrix_rank(A)
n = A.shape[1]

# nulidad = n - rango
nulidad = n - rango

print("Rango:", rango)
print("Nulidad:", nulidad)
print("Verificación:", rango + nulidad, "=", n)
</code></pre>
    <br>
    <h4>Ejercicio: experimentación con matrices aleatorias</h4>
    <pre><code class="language-python">
import numpy as np

def rango_y_nulidad(m, n):
    A = np.random.randn(m, n)
    rank = np.linalg.matrix_rank(A)
    null = n - rank
    return rank, null

rank, null = rango_y_nulidad(4, 6)
print("Rango:", rank, " | Nulidad:", null)
</code></pre>
  </article>
    <br>
  <footer>
    <h3>Actividades y tareas</h3>
    <ol>
      <li>Escribe un algoritmo que determine si un conjunto de vectores forma una base de \(\mathbb{R}^n\).</li>
      <li>Implementa un cambio de base genérico entre dos bases aleatorias de \(\mathbb{R}^3\).</li>
      <li>Genera ejemplos numéricos donde se verifique el teorema rango + nulidad.</li>
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