# -*- coding: utf-8 -*-
"""
Created on Thu May 20 12:17:47 2024

@author: lukas
"""

# import the system to simulate
from SystemsOld import HeatSystem

import PSO.swarmlip as swl
import mysql.connector
from mysql.connector import Error


def outputSwarm(swarm, particle_space):
    print()
    print("Position")
    print(swarm.gBest.pBest_position)
    print("ITAE")
    print(swarm.gBest.pBest_cost)
    print("W")
    print(swarm.options[0])
    print("Particle Space in %")
    print(particle_space)


def main():
    try:
        # database connection
        connection = mysql.connector.connect(
            host='127.0.0.1',  # host
            port='3306',  # port
            database='pid_data',  # database name
            user='root',  # database user
            password='Test123'  # password
        )

        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Verbunden zum MySQL Server in Version ", db_Info)
            cursor = connection.cursor()

            # Create table entry
            insert_query = """
            INSERT INTO pso_heatsystem (Kp, Ti, Td, itae, iterations, maxStall, space_precision, stall_precision)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            obj_func = HeatSystem.systemResponseX
            swarm_size = 40
            bounds = [[0, 0.1, 0], [100, 10000, 10]]
            print("STARTING PSO")
            # 100 simulations
            for i in range(100):
                swarm = swl.Swarm(obj_func, swarm_size, bounds)
                terminated_swarm = swarm.simulate_swarm(outputSwarm)
                Kp = terminated_swarm.gBest.pBest_position[0]
                Ki = terminated_swarm.gBest.pBest_position[1]
                Kd = terminated_swarm.gBest.pBest_position[2]
                itae = terminated_swarm.gBest.pBest_cost
                iterations = terminated_swarm.iterations
                maxStall = terminated_swarm.maxStall
                space_precision = terminated_swarm.spaceFactor
                stall_precision = terminated_swarm.convergenceFactor
                values = (Kp, Ki, Kd, itae, iterations, maxStall, space_precision, stall_precision)
                cursor.execute(insert_query, values)
                connection.commit()
                print("Eintrag erfolgreich hinzugefügt.")

    except Error as e:
        print("Fehler beim Verbinden zur MySQL", e)
    finally:
        # close connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL Verbindung ist geschlossen")


if __name__ == '__main__':
    main()
