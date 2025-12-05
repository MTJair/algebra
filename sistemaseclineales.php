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
<section id="tema3">
  <h1>Tema 3 — Sistemas de ecuaciones lineales</h1>
  <p>Este tema presenta una explicación completa y didáctica sobre sistemas lineales: definición del conjunto solución, representación matricial, sistemas equivalentes y clasificación según el tipo de solución. Incluye ejemplos resueltos, código en <code>Python</code> (listo para Spyder) y visualizaciones para apoyar la comprensión.</p>

  <article id="3-1">
    <h2>3.1 Conjunto solución</h2>
    <h4>Definición</h4>
    <p>Un <strong>sistema de ecuaciones lineales</strong> con m ecuaciones y n incógnitas es
      <br><em>\(\begin{cases} a_{11}x_1 + a_{12}x_2 + \dots + a_{1n}x_n = b_1 \\ a_{21}x_1 + a_{22}x_2 + \dots + a_{2n}x_n = b_2 \\ \vdots \\ a_{m1}x_1 + a_{m2}x_2 + \dots + a_{mn}x_n = b_m \end{cases}\)</em>.
    </p>
    <p>El <strong>conjunto solución</strong> es el conjunto de vectores \(x=(x_1,\dots,x_n)^T\) que satisfacen todas las ecuaciones simultáneamente. Puede ser:
      <ul>
        <li>vacío (ninguna solución),</li>
        <li>un único vector (solución única),</li>
        <li>o un conjunto infinito de soluciones (dependencia entre ecuaciones).</li>
      </ul>
    </p>

    <h4>Ejemplo simple</h4>
    <p>Considera el sistema de 2 ecuaciones y 2 incógnitas:</p>
    <blockquote>\(\begin{cases} x + y = 3 \\ 2x - y = 0 \end{cases}\)</blockquote>
    <p>Solución: \(x=1, y=2\), entonces el conjunto solución es el conjunto { (1, 2) }.</p>
  </article>

  <article id="3-2">
    <h2>3.2 Representación matricial</h2>
    <h4>Notación matricial</h4>
    <p>Podemos escribir el sistema como <strong>Ax = b</strong>, donde:</p>
    <ul>
      <li>A ∈ ℝ<sup>m×n</sup> es la matriz de coeficientes \(a_{ij}\).</li>
      <li>x ∈ ℝ<sup>n</sup> es el vector columna de incógnitas \(x_j\).</li>
      <li>b ∈ ℝ<sup>m</sup> es el vector de términos independientes \(b_i\).</li>
    </ul>

    <br>
    <h4>Interpretación</h4>
    <p>La ecuación Ax=b puede interpretarse como la combinación lineal de las columnas de A con coeficientes los componentes de x produciendo el vector b. Esto relaciona el problema de resolver el sistema con la teoría de subespacios y combinación lineal.</p>

    <h4>Ejemplo en Python (resolución directa)</h4>
    <br>
    <pre><code class="language-python"># Resolver Ax=b con numpy (caso único)
import numpy as np

A = np.array([[1, 1], [2, -1]], dtype=float)
b = np.array([3, 0], dtype=float)

x = np.linalg.solve(A, b)  # solo si A es cuadrada e invertible
print('Solución x =', x)
</code></pre>

    <br>
    <h4>Augmented matrix (matriz aumentada)</h4>
    <p>La matriz aumentada [A|b] se usa en eliminación gaussiana para aplicar operaciones elementales de fila con el objetivo de resolver el sistema.</p>
  </article>

  <article id="3-3">
    <h2>3.3 Sistemas equivalentes</h2>
    <h4>Definición</h4>
    <p>Dos sistemas son <strong>equivalentes</strong> si tienen el mismo conjunto solución. Las operaciones elementales de fila (intercambio de filas, multiplicar una fila por un escalar no nulo y sumar a una fila un múltiplo de otra fila) transforman un sistema en otro equivalente.</p>

    <h4>Eliminación Gaussiana (procedimiento)</h4>
    <br>
    <ol>
      <li>Construir la matriz aumentada [A|b].</li>
      <li>Aplicar operaciones elementales de fila para llevar la matriz a forma escalonada por filas.</li>
      <li>Usar sustitución hacia atrás si la matriz es cuadrada y de rango completo para encontrar la solución única.</li>
    </ol>

    <br>
    <h4>Implementación pedagógica en Python (eliminación Gaussiana simple)</h4>
    <br>
    <pre><code class="language-python"># Implementación simple de eliminación Gaussiana (no optimizada)
import numpy as np

def gaussian_elimination(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    m, n = A.shape
    # matriz aumentada
    M = np.hstack([A, b.reshape(-1,1)])

    row = 0
    for col in range(n):
        # buscar pivote
        pivot = None
        for r in range(row, m):
            if abs(M[r, col]) > 1e-12:
                pivot = r
                break
        if pivot is None:
            continue
        # intercambiar filas
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        # normalizar fila
        M[row] = M[row] / M[row, col]
        # eliminar abajo
        for r in range(row+1, m):
            M[r] = M[r] - M[r, col] * M[row]
        row += 1
        if row == m:
            break
    return M

# Ejemplo:
A = np.array([[1,1,1],[2,-1,1],[3,2,-1]], dtype=float)
b = np.array([6,1,2], dtype=float)
M_e = gaussian_elimination(A, b)
print('Matriz en forma escalonada (aumentada):')
print(M_e)
</code></pre>

    <br>
    <h4>Sustitución hacia atrás (cuando hay pivotes en cada variable)</h4>
    <br>
    <pre><code class="language-python"># Suponiendo que gaussian_elimination devolvió una matriz M escalonada completa
def back_substitution(M):
    # M: matriz aumentada en escalonada (m x (n+1)) con pivotes 1 en columna de pivote
    m, n1 = M.shape
    n = n1 - 1
    x = np.zeros(n)
    for i in reversed(range(n)):
        # buscar primer coeficiente no nulo en la fila i
        row = i
        # si fila de ceros, saltar (variables libres)
        pivot_col = None
        for j in range(n):
            if abs(M[row, j]) > 1e-12:
                pivot_col = j
                break
        if pivot_col is None:
            continue
        x[pivot_col] = M[row, -1] - np.dot(M[row, pivot_col+1:n], x[pivot_col+1:n])
    return x

# Verificación (usar con cuidado según la forma escalonada)
# x = back_substitution(M_e)
# print('Solución por sustitución hacia atrás:', x)
</code></pre>
  </article>

  <br>
  <article id="3-4">
    <h2>3.4 Tipos de sistemas según el conjunto solución</h2>

    <h4>Clasificación por rangos (criterio de Rouché–Capelli)</h4>
    <p>Sea A la matriz de coeficientes y [A|b] la matriz aumentada. Entonces:</p>
    <ul>
      <li>Si <strong>rank(A) = rank([A|b]) = n</strong> (n = número de incógnitas) ⇒ solución única.</li>
      <li>Si <strong>rank(A) = rank([A|b]) &lt; n</strong> ⇒ infinitas soluciones (hay variables libres).</li>
      <li>Si <strong>rank(A) &lt; rank([A|b])</strong> ⇒ sistema inconsistente (sin solución).</li>
    </ul>

    <br>
    <h4>Interpretación geométrica</h4>
    <p>En ℝ^2, dos rectas pueden intersectar en un punto (solución única), ser paralelas (sin solución) o coincidir (infinitas soluciones). En dimensiones mayores la intuición es análoga con planos e hiperplanos.</p>

    <h4>Ejemplos en Python</h4>
    <br>
    <pre><code class="language-python"># Ejemplos: sistema con solución única, sin solución e infinitas soluciones
import numpy as np

# 1) solución única
A1 = np.array([[1,1],[2,-1]], dtype=float)
b1 = np.array([3,0], dtype=float)
print('rank(A1)=', np.linalg.matrix_rank(A1), 'rank([A1|b1])=', np.linalg.matrix_rank(np.column_stack([A1,b1])))
print('Solución (numpy):', np.linalg.solve(A1,b1))

# 2) sin solución (rectas paralelas)
A2 = np.array([[1, -1],[2, -2]], dtype=float)
b2 = np.array([1, 3], dtype=float)
print('\nrank(A2)=', np.linalg.matrix_rank(A2), 'rank([A2|b2])=', np.linalg.matrix_rank(np.column_stack([A2,b2])))
# no usar np.linalg.solve aquí (A2 es singular)

# 3) infinitas soluciones (misma recta)
A3 = np.array([[1, -1],[2, -2]], dtype=float)
b3 = np.array([1, 2], dtype=float)
print('\nrank(A3)=', np.linalg.matrix_rank(A3), 'rank([A3|b3])=', np.linalg.matrix_rank(np.column_stack([A3,b3])))
# encontrar soluciones: paramétricas
# Resolución con numpy.linalg.lstsq (solución particular mínima) y nullspace para la parte homogénea
x_part, residuals, rank, s = np.linalg.lstsq(A3, b3, rcond=None)
print('Solución particular (mínima-norma):', x_part)

# Nullspace (espacio nulo) por SVD
U, S, Vt = np.linalg.svd(A3)
null_mask = (S <= 1e-12)
# una forma habitual para nullspace:
rankA3 = np.linalg.matrix_rank(A3)
ns = Vt[rankA3:].T
print('Bases del espacio nulo (aprox):', ns)
</code></pre>

    <br>
    <h4>Gráfica en 2D (intersección de rectas)</h4>
    <br>
    <pre><code class="language-python"># Graficar tres casos en 2D: única, sin solución y infinitas
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 4, 200)

# caso 1 (única): x + y = 3  y 2x - y = 0
y1 = 3 - x
y2 = 2*x
plt.figure(figsize=(6,4))
plt.plot(x, y1, label='x + y = 3')
plt.plot(x, y2, label='2x - y = 0')
plt.title('Intersección: solución única')
plt.legend()
plt.grid(True)
plt.ylim(-1,5)
plt.show()

# caso 2 (paralelas, sin solución): x - y = 1  y 2x - 2y = 3
y1 = x - 1
y2 = x - 1.5
plt.figure(figsize=(6,4))
plt.plot(x, y1, label='x - y = 1')
plt.plot(x, y2, label='2x - 2y = 3')
plt.title('Rectas paralelas: sin solución')
plt.legend()
plt.grid(True)
plt.ylim(-1,5)
plt.show()

# caso 3 (mismas rectas, infinitas soluciones): x - y = 1  y 2x - 2y = 2
y1 = x - 1
y2 = x - 1
plt.figure(figsize=(6,4))
plt.plot(x, y1, label='x - y = 1')
plt.plot(x, y2, label='2x - 2y = 2')
plt.title('Rectas coincidentes: infinitas soluciones')
plt.legend()
plt.grid(True)
plt.ylim(-1,5)
plt.show()
</code></pre>

    <br>
    <h4>Soluciones generales (forma paramétrica)</h4>
    <p>Cuando hay infinitas soluciones (rank(A) &lt; n) se puede escribir la solución general como:</p>
    <blockquote>\( x = x_p + t_1 v_1 + \dots + t_k v_k, \)</blockquote>
    <p>donde \(x_p\) es una solución particular, los \(v_i\) forman una base del espacio nulo de A, y los \(t_i\) son parámetros reales libres.</p>
  </article>

  <footer>
    <h3>Actividades y tareas</h3>
    <ol>
      <li>Implementa la eliminación Gaussiana con pivoteo parcial y compara con <code>np.linalg.solve</code> para sistemas cuadrados.</li>
      <li>Para matrices aleatorias de tamaño n×n, calcula numéricamente la probabilidad (medida empírica) de singularidad bajo distintos rangos de valores.</li>
      <li>Investiga la relación entre condicionamiento numérico de A (número de condición) y la sensibilidad de la solución a pequeñas perturbaciones en b.</li>
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