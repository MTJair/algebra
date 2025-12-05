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
<section id="tema6">
  <h1>Tema 6 — Aplicaciones lineales</h1>
  <p>Este tema presenta conceptos fundamentales sobre aplicaciones lineales: cómo se representan mediante matrices, cómo estudiar su núcleo e imagen y cómo se relacionan los cambios de base con la equivalencia y similitud de matrices.  
  Incluye ejemplos resueltos, código en <code>Python</code> y pequeños experimentos numéricos para profundizar en los conceptos.</p>

  <!-- 6.1 ----------------------------------------------------------------------------- -->
  <article id="6-1">
    <h2>6.1 Representación matricial de una aplicación lineal</h2>

    <h4>Definición</h4>
    <p>Sea \(T: \mathbb{R}^n \rightarrow \mathbb{R}^m\) una <strong>aplicación lineal</strong>.  
    Esto significa que para todo \(u,v\in\mathbb{R}^n\) y para todo escalar \(\alpha\), se cumple:
      <ul>
        <li>\(T(u+v)=T(u)+T(v)\)</li>
        <li>\(T(\alpha u)=\alpha T(u)\)</li>
      </ul>
    </p>

    <h4>Teorema fundamental</h4>
    <p>Toda aplicación lineal queda completamente determinada por su acción sobre los vectores de una base.  
    Si \(e_1,\dots,e_n\) es la base estándar, la matriz asociada a T es:
    <blockquote>\( A = [T(e_1)\ T(e_2)\ \dots\ T(e_n)] \)</blockquote>
    Así, para cualquier vector \(x\in\mathbb{R}^n\),
    <blockquote>\( T(x) = A x. \)</blockquote>
    </p>

    <h4>Ejemplo simple</h4>
    <p>Sea \(T(x,y) = (2x + y,\ -x + 3y)\). Su matriz es:
    <blockquote>
      \( T(e_1) = (2,-1)^T,\ T(e_2) = (1,3)^T \)  
      \(\Rightarrow A = \begin{pmatrix}2 & 1 \\ -1 & 3\end{pmatrix} \)
    </blockquote>
    </p>

    <h4>Ejemplo en Python</h4>
    <pre><code class="language-python">
import numpy as np

# T(x,y) = (2x + y, -x + 3y)
A = np.array([[2, 1],
              [-1, 3]], float)

x = np.array([4, -2])
print("T(x) = A x =", A @ x)
</code></pre>

    <br>
    <h4>Ejercicio en Python</h4>
    <p>Implementa una función que reciba T definida sobre los vectores de la base estándar y construya la matriz correspondiente.</p>
    <pre><code class="language-python">
import numpy as np

def matriz_de_T(vectores_imagen):
    # vectores_imagen es una lista [T(e1), T(e2), ...]
    return np.column_stack(vectores_imagen)

T_e1 = np.array([2, -1])
T_e2 = np.array([1, 3])

A = matriz_de_T([T_e1, T_e2])
print("Matriz de T:\n", A)
</code></pre>
  </article>

  <br>
  <!-- 6.2 ----------------------------------------------------------------------------- -->
  <article id="6-2">
    <h2>6.2 Núcleo e imagen</h2>

    <h4>Núcleo (kernel)</h4>
    <p>El <strong>núcleo</strong> de una aplicación lineal \(T\) es:
    <blockquote>
      \( \ker(T) = \{x \in \mathbb{R}^n : T(x)=0 \} \)
    </blockquote>
    Es un subespacio de \(\mathbb{R}^n\).</p>

    <h4>Imagen (range)</h4>
    <p>La <strong>imagen</strong> de \(T\) es:
    <blockquote>
      \( \text{Im}(T) = \{T(x): x\in\mathbb{R}^n\} \)
    </blockquote>
    Equivale al espacio columna de la matriz asociada.</p>

    <h4>Relación fundamental</h4>
    <p>Para \(A\) de tamaño \(m\times n\):
      <blockquote> \(\dim(\ker A) + \text{rank}(A) = n.\)</blockquote>
    </p>

    <h4>Ejemplo numérico</h4>
    <pre><code class="language-python">
import numpy as np

A = np.array([[1,2,3],
              [2,4,6],
              [1,0,1]], float)

# rango (dim imagen)
rango = np.linalg.matrix_rank(A)

# nulidad (dim núcleo)
nulidad = A.shape[1] - rango

print("Rango =", rango)
print("Nulidad =", nulidad)
</code></pre>

    <br>
    <h4>Ejemplo: espacio nulo por SVD</h4>
    <pre><code class="language-python">
def nullspace(A, tol=1e-12):
    U, S, Vt = np.linalg.svd(A)
    r = np.sum(S > tol)
    return Vt[r:].T

ns = nullspace(A)
print("Base del núcleo:\n", ns)
</code></pre>

    <br>
    <h4>Ejercicio en Python</h4>
    <p>Construye un script que, dada una matriz A, imprima:
      <ul>
        <li>una base del núcleo,</li>
        <li>una base del espacio columna,</li>
        <li>verifique que rango + nulidad = número de columnas.</li>
      </ul>
    </p>

    <pre><code class="language-python">
def bases_de_A(A):
    # base del espacio columna
    U, S, Vt = np.linalg.svd(A)
    r = np.sum(S > 1e-12)
    base_col = U[:, :r]

    # base del núcleo
    base_null = Vt[r:].T
    return base_col, base_null

base_col, base_null = bases_de_A(A)
print("Base del espacio columna:\n", base_col)
print("Base del núcleo:\n", base_null)
print("Verificación:", base_col.shape[1] + base_null.shape[1], "=", A.shape[1])
</code></pre>
  </article>

  <br>

  <!-- 6.3 ----------------------------------------------------------------------------- -->
  <article id="6-3">
    <h2>6.3 Cambios de base, matrices equivalentes y matrices similares</h2>

    <h4>Cambio de base</h4>
    <p>Sea \(B=\{b_1,\dots,b_n\}\) una base de \(\mathbb{R}^n\). Si un vector \(x\) tiene coordenadas \([x]_B\), entonces:
      <blockquote>
        \( x = P_B [x]_B \),  
      </blockquote>
      donde \(P_B = [\,b_1\ b_2\ \dots\ b_n\,]\) es la matriz de cambio de base.</p>

    <h4>Transformación de una aplicación bajo un cambio de base</h4>
    <p>Si la matriz de T en la base estándar es A, entonces en la base B es:
      <blockquote>
        \( A_B = P_B^{-1} A P_B.\)
      </blockquote>
    </p>

    <h4>Matrices equivalentes</h4>
    <p>Dos matrices A y B son <strong>equivalentes</strong> si se obtienen mediante operaciones elementales de fila o columna.  
    Relacionado con clasificación por rango.</p>

    <h4>Matrices similares</h4>
    <p>Dos matrices A y B son <strong>similares</strong> si existe una matriz invertible P tal que:
      <blockquote> \( B = P^{-1} A P. \)</blockquote>
    Las matrices similares representan la misma aplicación lineal en distintas bases.</p>

    <h4>Ejemplo en Python: cambio de base</h4>
    <pre><code class="language-python">
import numpy as np

# matriz de T en base estándar
A = np.array([[3, 1],
              [0, 2]])

# nueva base B = {b1, b2}
b1 = np.array([1, 1])
b2 = np.array([1, -1])
P = np.column_stack([b1, b2])   # matriz cambio de base

# matriz de T en la base B
A_B = np.linalg.inv(P) @ A @ P
print("Matriz de T en base B:\n", A_B)
</code></pre>

    <br>
    <h4>Ejemplo: verificar similitud</h4>
    <pre><code class="language-python">
# verificar que A_B es similar a A
A_recuperada = P @ A_B @ np.linalg.inv(P)
print("Recuperación de A:\n", A_recuperada)
</code></pre>

    <br>
    <h4>Ejercicio en Python</h4>
    <p>Genera una matriz aleatoria A, escoge una base aleatoria (matriz invertible P) y verifica que:
      <ul>
        <li>A y \(P^{-1}AP\) tienen el mismo polinomio característico,</li>
        <li>tienen los mismos autovalores.</li>
      </ul>
    </p>

    <pre><code class="language-python">
import numpy as np

A = np.random.randn(3,3)
P = np.random.randn(3,3)

# asegurar invertibilidad
while np.linalg.det(P) == 0:
    P = np.random.randn(3,3)

B = np.linalg.inv(P) @ A @ P

# eigenvalores
eig_A = np.linalg.eigvals(A)
eig_B = np.linalg.eigvals(B)

print("Autovalores de A:", eig_A)
print("Autovalores de B:", eig_B)
</code></pre>
  </article>

  <br>

  <footer>
    <h3>Actividades y tareas</h3>
    <ol>
      <li>Construye una aplicación lineal T de tu elección, calcula su matriz en dos bases distintas y verifica la relación \(A_B = P^{-1}AP\).</li>
      <li>Genera matrices con distintos rangos y estudia cómo cambian sus núcleos e imágenes.</li>
      <li>Investiga numéricamente cuándo dos matrices aleatorias parecen ser similares (mismos eigenvalores) pero no son equivalentes.</li>
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